import subprocess
from onnxruntime.capi import _pybind_state as C

from inference.core.logger import logger

HAILO_DEVICES = [
    "hailo8",
    "hailo8r",
    "hailo8l",
    "hailo15h",
    "hailo15m",
    "hailo15l",
    "hailo10h",
    "pluto",
]
HAILO_DEVICES_DICT = dict(
    hailo8="HAILO-8 ",
    hailo8r="HAILO-8R",
    hailo8l="HAILO-8L",
    hailo15h="HAILO-15H",
    hailo15m="HAILO-15M",
    hailo15l="HAILO-15L",
    hailo10h="HAILO-10H",
    pluto="PLUTO",
)
DEFAULT_HW_ARCH = "hailo8"


def extract_board_config(board_config)->list:
    avable_device = []
    for device, device_config in HAILO_DEVICES_DICT.items():
        if device_config in board_config:
            avable_device.append(device)

    return avable_device


def get_hailo_devices()->list:
    try:
        from hailo_platform import VDevice
    except ImportError:
        logger.warning("Hailo SDK is not installed. Skipping Hailo conversion")
        return []
    # get hailo available devices
    hailo_available_devices = []
    try:
        physical_devices = VDevice().get_physical_devices()
    except Exception as e:
        logger.warning(f"Error getting hailo devices: {e}")
        return hailo_available_devices
    
    for device in physical_devices:
        try:
            board_config = str(device.control.read_board_config())

            hailo_available_devices += extract_board_config(board_config)
            hailo_available_devices = list(set(hailo_available_devices))

        except Exception as e:
            logger.warning(f"Error getting hailo devices: {e}")

    return hailo_available_devices


def get_all_devices()->list:
    # get onnx available devices
    onnx_available_devices = C.get_available_providers()

    # get hailo available devices
    hailo_available_devices = get_hailo_devices()
    return onnx_available_devices + hailo_available_devices

def check_hailo_device()->bool:
    output = subprocess.run("lspci",capture_output=True, check=True).stdout.decode()
    
    return "Hailo" in output

if __name__ == "__main__":
    print(check_hailo_device())
