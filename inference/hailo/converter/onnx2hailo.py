import os
import os.path as osp
import onnx
import numpy as np
from inference.core.logger import logger
from onnxruntime.capi import _pybind_state as C
from inference.hailo.utils.devices import get_optimal_providen


def generate_calibrat_datasets(
    image_dir=None, image_shape=(500, 640, 640, 3)
) -> np.ndarray:
    return np.random.randn(*image_shape).astype("f")


class OnnxRT2HailoRT:
    def __init__(self, onnx_file: str = None, providers=None, arch: str = "hailo8"):
        """
        Parameters
        ----------
        onnx_file: str
            Path to the onnx model file
        providers: List[str]
            List of providers to use for onnxruntime, default is None
        arch: str
            Target architecture name, default is "hailo8"

        Attributes
        ----------
        onnx_file: str
            Path to the onnx model file
        target_arch: str
            Target architecture name
        all_available_devices: List[str]
            List of all available Hailo devices
        """

        super().__init__()
        self.onnx_file = onnx_file
        self.target_arch = arch
        self.all_available_devices = get_optimal_providen()

    @property
    def hef_file(self):
        """
        Property to get the path to the hef file, which is the onnx file with the extension replaced with ".hef".

        Returns
        -------
        str
            Path to the hef file
        """
        return self.onnx_file.replace(".onnx", ".hef")

    @property
    def hailoonnx_file(self):
        """
        Property to get the path to the modified onnx file, which is the onnx file with "_hailo" added to the filename before the extension.

        Returns
        -------
        str
            Path to the modified onnx file
        """
        return self.onnx_file.replace(".onnx", "_hailo.onnx")

    def translate_onnx2hef(self) -> str:
        """
        Translate an onnx model to a hef model.

        This function translates an onnx model to a hef model using the Hailo SDK client.
        It first loads the onnx model, then generates calibration datasets, optimizes the model
        using the calibration datasets, compiles the model, saves the model as a hef file, and
        saves the model as an onnx file with "_hailo" added to the filename before the extension.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Path to the hef file
        """
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error(ie)
            logger.warning("Please install hailo-sdk-client to use this function.")

        runner = ClientRunner(hw_arch=self.target_arch)
        onnx_model_graph_info = self.get_onnx_info()
        net_input_shapes = {
            name: shape
            for name, shape in zip(
                onnx_model_graph_info["start_nodes_name"],
                onnx_model_graph_info["inputs_shape"],
            )
        }

        # load onnx model
        runner.translate_onnx_model(
            self.onnx_file,
            start_node_names=onnx_model_graph_info["start_nodes_name"],
            end_node_names=onnx_model_graph_info["end_nodes_name"],
            net_input_shapes=net_input_shapes,
        )

        # generate calibration datasets
        calibrat_datasets = generate_calibrat_datasets()

        # optimize model
        runner.optimize(calibrat_datasets)

        # compile model
        hef_model = runner.compile()

        # save hef model
        self.save_model(hef_model, self.hef_file)

        # save model
        onnx_model_for_hailo = runner.get_hailo_runtime_model()
        onnx.save(onnx_model_for_hailo, self.hailoonnx_file)

        return self.hef_file

    def save_model(self, model, file_path):
        """
        Save the model to a file.

        Parameters
        ----------
        model: bytes
            Hailo model in bytes
        file_path: str
            Path to the file to save the model
        """
        with open(file_path, "wb") as f:
            f.write(model)

    def get_onnx_info(self) -> dict:
        """
        Retrieve information about the ONNX model's input and output nodes.

        This function loads an ONNX model file, verifies its correctness, and extracts
        the shapes and names of the input and output nodes from the model's graph.

        Returns
        -------
        dict
            A dictionary containing:
            - "inputs_shape": List of tuples representing the shapes of input nodes.
            - "outputs_shape": List of tuples representing the shapes of output nodes.
            - "start_nodes_name": List of names of the input nodes.
            - "end_nodes_name": List of names of the output nodes.
        """

        onnx_model = onnx.load(self.onnx_file)
        onnx.checker.check_model(onnx_model)
        inputs = []
        start_nodes_name = []
        for i in onnx_model.graph.input:
            inputs.append(
                tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
            )
            start_nodes_name.append(i.name)

        outputs = []
        end_nodes_name = []
        for i in onnx_model.graph.output:
            outputs.append(
                tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
            )
            end_nodes_name.append(i.name)

        return {
            "inputs_shape": inputs,
            "outputs_shape": outputs,
            "start_nodes_name": start_nodes_name,
            "end_nodes_name": end_nodes_name,
        }

    def translate_onnx2hailoonnx(self):
        pass

    def check_onnx_backend(self):
        pass

    def transform(self):
        pass

    def save(self):
        pass
