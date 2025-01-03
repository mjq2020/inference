import onnx
import numpy as np
from inference.core.logger import logger
from onnxruntime.capi import _pybind_state as C
from inference.hailo.utils.devices import get_all_devices


def generate_calibrat_datasets(
    image_dir=None, image_shape=(500, 640, 640, 3)
) -> np.ndarray:
    return np.random.randn(*image_shape).astype("f")


class Onnx2HailoRunner:
    def __init__(self, onnx_path=None, providers=None, arch="hailo8"):
        super().__init__()
        self.onnx_path = onnx_path
        self.target_arch = arch
        self.all_available_devices = get_all_devices()

    def translate_onnx2hef(self) -> str:
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
            self.onnx_path,
            start_node_names=onnx_model_graph_info["start_nodes_name"],
            end_node_names=onnx_model_graph_info["end_nodes_name"],
            net_input_shapes=net_input_shapes,
        )

        # generate calibration datasets
        calibrat_datasets = generate_calibrat_datasets()

        # optimize model
        runner.optimize(calibrat_datasets)

        # compile model
        runner.compile()

        # save model
        onnx_model_for_hailo = runner.get_hailo_runtime_model()
        onnx.save(onnx_model_for_hailo, "onnx_model_for_hailo.onnx")

    def get_onnx_info(self)->dict:
        onnx_model = onnx.load(self.onnx_path)
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


if __name__ == "__main__":
    Onnx2HailoRunner(
        r"/home/dq/github/sscma/work_dirs/rtmdet_nano_8xb32_300e_coco_ncadc_relu6/epoch_300.onnx"
    ).translate_onnx2hef()
