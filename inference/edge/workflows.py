"""Bounded adapter around the project's original Workflow execution engine."""

import hashlib
import json
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from threading import RLock

from .errors import EdgeError
from .images import decode_image
from .limits import check_value, workflow_budget


class EdgeWorkflows:
    def __init__(self, model_manager, settings):
        self.manager = model_manager
        self.settings = settings
        self.definitions = None
        self._cache = OrderedDict()
        self._frame_numbers = {}
        self._descriptions = OrderedDict()
        self._schema = None
        self._lock = RLock()

    def describe(self, execution_engine_version=None, dynamic_blocks_definitions=None):
        """Return the original Builder's connection and query-language metadata."""
        if dynamic_blocks_definitions:
            raise EdgeError(
                "Custom Python blocks are unavailable in the RV1126B runtime",
                code="unsupported_workflow_block",
                status_code=422,
            )
        try:
            with self._lock:
                if execution_engine_version not in self._descriptions:
                    self._descriptions[execution_engine_version] = self._describe(
                        execution_engine_version
                    )
                    while len(self._descriptions) > 2:
                        self._descriptions.popitem(last=False)
                return self._descriptions[execution_engine_version]
        except Exception as exc:
            self._raise_workflow_error(exc)

    def _describe(self, execution_engine_version):
        from inference.core.entities.responses.workflows import (
            ExternalBlockPropertyPrimitiveDefinition,
            ExternalWorkflowsBlockSelectorDefinition,
            UniversalQueryLanguageDescription,
            WorkflowsBlocksDescription,
        )
        from inference.core.workflows.core_steps.common.query_language.introspection.core import (
            prepare_operations_descriptions,
            prepare_operators_descriptions,
        )
        from inference.core.workflows.execution_engine.introspection.blocks_loader import (
            describe_available_blocks,
        )
        from inference.core.workflows.execution_engine.introspection.connections_discovery import (
            discover_blocks_connections,
        )
        from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
            DynamicBlockDefinition,
        )

        description = describe_available_blocks(
            dynamic_blocks=[], execution_engine_version=execution_engine_version
        )
        connections = discover_blocks_connections(blocks_description=description)
        kinds = {
            name: [
                ExternalWorkflowsBlockSelectorDefinition(
                    **{
                        key: getattr(connection, key)
                        for key in ExternalWorkflowsBlockSelectorDefinition.model_fields
                    }
                )
                for connection in sorted(
                    values,
                    key=lambda item: (
                        item.manifest_type_identifier,
                        item.property_name,
                        item.compatible_element,
                    ),
                )
            ]
            for name, values in connections.kinds_connections.items()
        }
        primitives = [
            ExternalBlockPropertyPrimitiveDefinition(
                **{
                    key: getattr(connection, key)
                    for key in ExternalBlockPropertyPrimitiveDefinition.model_fields
                }
            )
            for connection in connections.primitives_connections
        ]
        dynamic_schema = DynamicBlockDefinition.model_json_schema()
        dynamic_schema.update(
            {"not": {}, "description": "Custom Python blocks are disabled on RV1126B."}
        )
        result = WorkflowsBlocksDescription(
            blocks=description.blocks,
            declared_kinds=description.declared_kinds,
            kinds_connections=kinds,
            primitives_connections=primitives,
            universal_query_language_description=UniversalQueryLanguageDescription.from_internal_entities(
                operations_descriptions=prepare_operations_descriptions(),
                operators_descriptions=prepare_operators_descriptions(),
            ),
            dynamic_block_definition_schema=dynamic_schema,
        ).model_dump(mode="json", by_alias=True)
        # Describe only the installed runtime's blocks. Keep the original model
        # type as the primary identifier, so existing Builder selectors work.
        for block, source in zip(result["blocks"], description.blocks):
            info = {"available": True}
            tasks = source.manifest_class.get_compatible_task_types()
            if tasks is not None:
                info["compatible_task_types"] = tasks
            block["block_schema"].setdefault("json_schema_extra", {})[
                "air_gapped_info"
            ] = info
        return result

    def schema(self):
        from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
            get_workflow_schema_description,
        )

        with self._lock:
            if self._schema is None:
                self._schema = get_workflow_schema_description().model_dump(
                    mode="json", by_alias=True
                )
            return self._schema

    def describe_interface(self, specification, api_key=None):
        from inference.core.workflows.execution_engine.v1.introspection.inputs_discovery import (
            describe_workflow_inputs,
        )
        from inference.core.workflows.execution_engine.v1.introspection.outputs_discovery import (
            describe_workflow_outputs,
        )
        from inference.core.workflows.execution_engine.v1.introspection.types_discovery import (
            discover_kinds_schemas,
            discover_kinds_typing_hints,
        )

        try:
            self._validate_shape(specification)
            with self._lock:
                engine = self._compile(specification, api_key=api_key)
                try:
                    parsed = engine._engine._compiled_workflow.workflow_definition
                    # The compiler has already expanded inner-workflow selectors.
                    # Describe that graph using the existing kind introspection.
                    expanded = {
                        "version": parsed.version,
                        "inputs": [
                            {
                                **item.model_dump(mode="json"),
                                "kind": [
                                    kind.name if hasattr(kind, "name") else kind
                                    for kind in item.kind
                                ],
                            }
                            for item in parsed.inputs
                        ],
                        "steps": [
                            step.model_dump(mode="json") for step in parsed.steps
                        ],
                        "outputs": [
                            output.model_dump(mode="json") for output in parsed.outputs
                        ],
                    }
                finally:
                    self._close_engine(engine)
                inputs = describe_workflow_inputs(definition=expanded)
                # The engine also permits direct input outputs. Upstream's
                # step-only introspector rejects those otherwise valid graphs.
                direct, step_outputs = {}, []
                for output in expanded["outputs"]:
                    selector = output.get("selector", "")
                    if selector.startswith("$inputs."):
                        direct[output["name"]] = inputs[selector[len("$inputs.") :]]
                    else:
                        step_outputs.append(output)
                outputs = describe_workflow_outputs(
                    definition={**expanded, "outputs": step_outputs}
                )
                outputs.update(direct)
                kinds = set()
                for value in [*inputs.values(), *outputs.values()]:
                    for names in value.values() if isinstance(value, dict) else [value]:
                        kinds.update(names)
                return {
                    "inputs": inputs,
                    "outputs": outputs,
                    "typing_hints": discover_kinds_typing_hints(kinds_names=kinds),
                    "kinds_schemas": discover_kinds_schemas(kinds_names=kinds),
                }
        except Exception as exc:
            self._raise_workflow_error(exc)

    def dynamic_outputs(self, step_manifest):
        """Parse a supported manifest without resolving selectors or loading NPU."""
        from inference.core.workflows.execution_engine.introspection.blocks_loader import (
            load_workflow_blocks,
        )
        from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
            parse_workflow_definition,
        )

        specification = {
            "version": "1.0",
            "inputs": [],
            "steps": [step_manifest],
            "outputs": [],
        }
        try:
            self._validate_shape(specification)
            with self._lock:
                parsed = parse_workflow_definition(
                    raw_workflow_definition=specification,
                    available_blocks=load_workflow_blocks(),
                )
                return [
                    output.model_dump(mode="json", by_alias=True)
                    for output in parsed.steps[0].get_actual_outputs()
                ]
        except Exception as exc:
            self._raise_workflow_error(exc)

    def _validate_shape(self, specification):
        if not isinstance(specification, dict):
            raise EdgeError(
                "Workflow specification must be an object",
                code="invalid_workflow",
                status_code=422,
            )
        for field in ("inputs", "steps", "outputs"):
            collection = specification.get(field, [])
            if not isinstance(collection, list) or any(
                not isinstance(item, dict) for item in collection
            ):
                raise EdgeError(
                    f"Workflow {field} must be a list of objects",
                    code="invalid_workflow",
                    status_code=422,
                )
        if len(specification.get("steps", [])) > self.settings.max_workflow_steps:
            raise EdgeError(
                "Workflow exceeds step budget",
                code="workflow_too_large",
                status_code=422,
            )
        from inference.core.workflows.core_steps.loader_rv1126b import load_blocks

        supported = {
            identifier
            for block in load_blocks()
            for identifier in block.get_manifest()
            .model_fields["type"]
            .annotation.__args__
        }
        for step in specification.get("steps", []):
            if step.get("type") not in supported:
                raise EdgeError(
                    "Workflow contains a block unavailable in the RV1126B runtime",
                    code="unsupported_workflow_block",
                    status_code=422,
                )
        if specification.get("dynamic_blocks_definitions"):
            raise EdgeError(
                "Custom Python blocks are unavailable in the RV1126B runtime",
                code="unsupported_workflow_block",
                status_code=422,
            )

    def _compile(
        self,
        specification,
        *,
        disable_sinks=False,
        workflow_id=None,
        profiler=None,
        api_key=None,
    ):
        from inference.core.workflows.execution_engine.core import ExecutionEngine
        from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
            normalize_inner_workflow_references_in_definition,
        )

        cloud_key = self.definitions.cloud_key(api_key) if self.definitions else None
        init_parameters = {
            "model_manager": self.manager,
            "api_key": cloud_key,
            "workflows_core.api_key": cloud_key,
            "disable_sinks": disable_sinks,
            "workflows_core.inner_workflow_spec_resolver": self._resolve_inner,
        }
        specification = normalize_inner_workflow_references_in_definition(
            specification, init_parameters
        )

        engine = ExecutionEngine.init(
            workflow_definition=specification,
            init_parameters=init_parameters,
            max_concurrent_steps=1,
            prevent_local_images_loading=True,
            workflow_id=workflow_id,
            profiler=profiler,
        )
        if (
            len(engine._engine._compiled_workflow.steps)
            > self.settings.max_workflow_steps
        ):
            self._close_engine(engine)
            raise EdgeError(
                "Expanded Workflow exceeds the configured step budget",
                code="workflow_too_large",
                status_code=422,
            )
        return engine

    def resolve_definition(self, specification, api_key=None):
        """Freeze referenced children before a long-lived video session starts."""
        from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
            normalize_inner_workflow_references_in_definition,
        )

        cloud_key = self.definitions.cloud_key(api_key) if self.definitions else None
        return normalize_inner_workflow_references_in_definition(
            specification,
            {
                "workflows_core.inner_workflow_spec_resolver": self._resolve_inner,
                "workflows_core.api_key": cloud_key,
            },
        )

    def _resolve_inner(
        self, workspace, workflow_id, workflow_version_id, init_parameters
    ):
        if self.definitions is None:
            raise EdgeError(
                "Saved inner Workflows require a Workflow definition registry",
                code="workflow_registry_unavailable",
                status_code=503,
            )
        return self.definitions.resolve(
            workspace,
            workflow_id,
            api_key=init_parameters.get("workflows_core.api_key"),
            workflow_version_id=workflow_version_id,
        )

    @staticmethod
    def _close_engine(engine):
        """Release block-owned caches deterministically at the session boundary."""
        compiled = getattr(getattr(engine, "_engine", None), "_compiled_workflow", None)
        failures = []
        for initialised in getattr(compiled, "steps", {}).values():
            close = getattr(initialised.step, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise EdgeError(
                "Workflow block cleanup failed",
                code="workflow_cleanup_failed",
                status_code=503,
            ) from failures[0]

    @staticmethod
    def _raise_workflow_error(exc):
        from inference.core.workflows.errors import (
            NotSupportedExecutionEngineError,
            RuntimeInputError,
            WorkflowCompilerError,
            WorkflowEnvironmentConfigurationError,
            WorkflowError,
            WorkflowExecutionEngineVersionError,
        )

        # The original scheduler wraps block failures. Preserve model/NPU errors
        # and their HTTP status through any number of Workflow wrapper layers.
        cause = exc
        edge_cause = None
        seen = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, EdgeError):
                if not isinstance(exc, WorkflowError):
                    raise cause
                edge_cause = cause
                break
            cause = getattr(cause, "inner_error", None) or cause.__cause__
        if isinstance(exc, WorkflowError):
            from inference.core.entities.responses.workflows import (
                WorkflowErrorResponse,
            )
            from inference.core.workflows.errors import (
                StepExecutionError,
                WorkflowBlockError,
            )

            blocks = getattr(exc, "blocks_errors", None)
            if isinstance(exc, StepExecutionError):
                blocks = [
                    WorkflowBlockError(
                        block_id=exc.block_id,
                        block_type=exc.block_type,
                        property_details=(
                            str(exc.inner_error) if exc.inner_error else None
                        ),
                        block_traceback=exc.block_traceback,
                    )
                ]
            response = WorkflowErrorResponse(
                message=exc.public_message,
                error_type=type(exc).__name__,
                context=exc.context,
                inner_error_type=exc.inner_error_type,
                inner_error_message=str(exc.inner_error) if exc.inner_error else None,
                blocks_errors=blocks,
                python_blocks_output_streams=getattr(
                    exc, "python_blocks_output_streams", None
                ),
                python_blocks_debug_traces=getattr(
                    exc, "python_blocks_debug_traces", None
                ),
            ).model_dump(mode="json")
            client_error = isinstance(
                exc,
                (
                    WorkflowCompilerError,
                    NotSupportedExecutionEngineError,
                    RuntimeInputError,
                    WorkflowExecutionEngineVersionError,
                ),
            )
            error = edge_cause or EdgeError(
                exc.public_message,
                code=(
                    "invalid_workflow" if client_error else "workflow_execution_failed"
                ),
                status_code=getattr(exc, "status_code", 400 if client_error else 500),
            )
            response["error"] = error.code
            error.workflow_response = response
            raise error from exc
        if isinstance(
            exc,
            (
                WorkflowCompilerError,
                WorkflowEnvironmentConfigurationError,
                WorkflowExecutionEngineVersionError,
                NotSupportedExecutionEngineError,
                RuntimeInputError,
                ValueError,
                TypeError,
                KeyError,
            ),
        ):
            raise EdgeError(
                str(exc)[:2000], code="invalid_workflow", status_code=422
            ) from exc
        raise exc

    def preflight_models(self, specification, inputs=None):
        parameters = {
            item["name"]: item["default_value"]
            for item in specification.get("inputs", [])
            if "name" in item and "default_value" in item
        }
        parameters.update(inputs or {})
        for step in specification.get("steps", []):
            if "model_id" not in step:
                continue
            identifier = step["model_id"]
            if isinstance(identifier, str) and identifier.startswith("$inputs."):
                identifier = parameters.get(identifier[8:])
            if not isinstance(identifier, str) or identifier.startswith("$"):
                raise EdgeError("Set the Workflow model input before starting inference.", code="model_input_required")
            self.manager.store.get(identifier, verify=False)

    def validate(self, specification, api_key=None):
        """Compile before opening a camera; this never loads model weights."""
        try:
            self._validate_shape(specification)
            with self._lock:
                self._close_engine(self._compile(specification, api_key=api_key))
        except Exception as exc:
            self._raise_workflow_error(exc)

    def run_request(
        self,
        specification,
        inputs,
        *,
        enable_profiling=False,
        debug=False,
        is_preview=False,
        disable_sinks=False,
        workflow_id=None,
        excluded_fields=None,
        api_key=None,
    ):
        """Original HTTP execution options, using the original profiler/debug contexts."""
        from inference.core.env import ENABLE_WORKFLOWS_PROFILING
        from inference.core.workflows.execution_engine.profiling.core import (
            BaseWorkflowsProfiler,
            NullWorkflowsProfiler,
        )
        from inference.core.workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
            register_debug_session,
        )

        profiler = (
            BaseWorkflowsProfiler
            if enable_profiling and ENABLE_WORKFLOWS_PROFILING
            else NullWorkflowsProfiler
        ).init(max_runs_in_buffer=1)
        with register_debug_session() if debug else nullcontext() as session:
            try:
                outputs = self._run(
                    specification,
                    inputs,
                    stream_id=None,
                    frame_metadata=None,
                    disable_sinks=disable_sinks,
                    is_preview=is_preview,
                    workflow_id=workflow_id,
                    profiler=profiler,
                    api_key=api_key,
                )
            except Exception as exc:
                if session is not None:
                    exc.python_blocks_output_streams = (
                        session.output_streams.snapshot() or None
                    )
                    exc.python_blocks_debug_traces = (
                        session.debug_traces.snapshot() or None
                    )
                self._raise_workflow_error(exc)
            with profiler.profile_execution_phase(
                name="workflow_results_filtering",
                categories=["inference_package_operation"],
            ):
                if excluded_fields:
                    excluded = set(excluded_fields)
                    outputs = [
                        {k: v for k, v in item.items() if k not in excluded}
                        for item in outputs
                    ]
            return {
                "outputs": outputs,
                "profiler_trace": profiler.export_trace(),
                "python_blocks_output_streams": (
                    session.output_streams.snapshot() or None if session else None
                ),
                "python_blocks_debug_traces": (
                    session.debug_traces.snapshot() or None if session else None
                ),
            }

    def run(
        self,
        specification,
        inputs,
        *,
        stream_id=None,
        frame_metadata=None,
        video_metadata_input_name="video_metadata",
        disable_sinks=False,
        is_preview=False,
        workflow_id=None,
        api_key=None,
    ):
        try:
            return self._run(
                specification,
                inputs,
                stream_id=stream_id,
                frame_metadata=frame_metadata,
                video_metadata_input_name=video_metadata_input_name,
                disable_sinks=disable_sinks,
                is_preview=is_preview,
                workflow_id=workflow_id,
                api_key=api_key,
            )
        except Exception as exc:
            self._raise_workflow_error(exc)

    def _run(
        self,
        specification,
        inputs,
        *,
        stream_id,
        frame_metadata,
        video_metadata_input_name="video_metadata",
        disable_sinks=False,
        is_preview=False,
        workflow_id=None,
        profiler=None,
        api_key=None,
    ):
        self._validate_shape(specification)
        if not isinstance(inputs, dict):
            raise EdgeError(
                "Workflow inputs must be an object",
                code="invalid_workflow",
                status_code=422,
            )
        # HTTP requests receive a fresh engine, so ContinueIf's delay state
        # cannot leak between callers. Only video sessions retain an engine.
        key = None
        if stream_id is not None:
            key = (
                stream_id,
                hashlib.sha256(
                    json.dumps(specification, sort_keys=True).encode()
                ).hexdigest(),
                disable_sinks,
                workflow_id,
                hashlib.sha256(api_key.encode()).hexdigest() if api_key else None,
            )
        with self._lock:
            metadata = None
            if stream_id is not None:
                metadata = {
                    "video_identifier": stream_id,
                    "frame_number": self._frame_numbers.get(stream_id, 1),
                    "frame_timestamp": datetime.now(timezone.utc),
                    "comes_from_video_file": False,
                    **(frame_metadata or {}),
                }
                metadata["video_identifier"] = stream_id
            parameters = dict(inputs)
            total_pixels = 0

            def decode_batch(value):
                nonlocal total_pixels
                if isinstance(value, list):
                    return [decode_batch(element) for element in value]
                array = decode_image(
                    value,
                    max_pixels=self.settings.max_image_pixels,
                    max_bytes=self.settings.max_request_bytes,
                )
                total_pixels += array.shape[0] * array.shape[1]
                if total_pixels > self.settings.max_image_pixels:
                    raise EdgeError(
                        "Combined Workflow images exceed the pixel budget",
                        code="image_too_large",
                        status_code=413,
                    )
                if isinstance(value, dict) or metadata is not None:
                    result = {
                        **(value if isinstance(value, dict) else {}),
                        "type": "numpy",
                        "value": array,
                    }
                    if metadata is not None:
                        result["video_metadata"] = metadata
                    return result
                return array

            for declared in specification.get("inputs", []):
                image_input = declared.get("type") in (
                    "WorkflowImage",
                    "InferenceImage",
                )
                kinds = declared.get("kind", [])
                if declared.get("type") == "WorkflowBatchInput":
                    image_input = any(
                        (kind.get("name") if isinstance(kind, dict) else kind)
                        == "image"
                        for kind in kinds
                    )
                if image_input:
                    name = declared.get("name")
                    if name not in parameters:
                        continue  # The original validator reports missing inputs.
                    parameters[name] = decode_batch(parameters[name])
                elif (
                    declared.get("type") == "WorkflowVideoMetadata"
                    and metadata is not None
                    and declared.get("name") == video_metadata_input_name
                ):
                    parameters[video_metadata_input_name] = metadata
            engine = self._cache.pop(key, None) if key else None
            if engine is None:
                engine = self._compile(
                    specification,
                    disable_sinks=disable_sinks,
                    workflow_id=workflow_id,
                    profiler=profiler,
                    api_key=api_key,
                )
            if key:
                self._cache[key] = engine
                while len(self._cache) > self.settings.workflow_cache_size:
                    old_key = next(iter(self._cache))
                    self._close_engine(self._cache[old_key])
                    del self._cache[old_key]
                    if not any(k[0] == old_key[0] for k in self._cache):
                        self._frame_numbers.pop(old_key[0], None)
            try:
                with workflow_budget(self.settings, video=stream_id is not None):
                    outputs = engine.run(
                        runtime_parameters=parameters,
                        fps=(metadata or {}).get("fps") or 0,
                        serialize_results=True,
                        _is_preview=is_preview,
                    )
                    check_value(outputs)
            finally:
                if key is None:
                    self._close_engine(engine)
            if stream_id is not None:
                self._frame_numbers[stream_id] = metadata["frame_number"] + 1
            return outputs

    def clear(self, stream_id=None):
        with self._lock:
            if stream_id is None:
                for engine in self._cache.values():
                    self._close_engine(engine)
                self._cache.clear()
                self._frame_numbers.clear()
            else:
                for key in list(self._cache):
                    if key[0] == stream_id:
                        self._close_engine(self._cache[key])
                        del self._cache[key]
                self._frame_numbers.pop(stream_id, None)
