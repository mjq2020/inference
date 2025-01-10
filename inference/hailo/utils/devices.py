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


def extract_board_config(board_config) -> list:
    """
    Extracts and returns a list of available Hailo devices present in the given board configuration.

    Args:
        board_config (str): A string containing the board configuration details.

    Returns:
        list: A list of strings representing the available Hailo devices identified in the board configuration.
    """

    avable_device = []
    for device, device_config in HAILO_DEVICES_DICT.items():
        if device_config in board_config:
            avable_device.append(device)

    return avable_device


def get_hailo_devices() -> list:
    """
    Returns a list of available Hailo devices on the system.

    The function first checks if the Hailo SDK is installed. If not, it logs a warning message and returns an empty list.

    It then attempts to get the physical devices using the Hailo SDK. If an exception occurs during this step, it logs a warning message and returns an empty list.

    For each physical device, it reads the board configuration and extracts the available Hailo devices using the `extract_board_config` function. If an exception occurs during this step, it logs a warning message and continues to the next device.

    Finally, it returns the list of available Hailo devices.

    Returns:
        list: A list of strings representing the available Hailo devices on the system.
    """
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


def get_all_devices() -> list:
    # get onnx available devices
    """
    Retrieves a list of all available devices for inference.

    It combines available devices from both ONNX and Hailo platforms.
    The function first obtains the list of available ONNX providers.
    Then, it retrieves the available Hailo devices using the `get_hailo_devices` function.

    Returns:
        list: A combined list of strings representing the available devices on both ONNX and Hailo platforms.
    """

    onnx_available_devices = C.get_available_providers()

    # get hailo available devices
    hailo_available_devices = get_hailo_devices()
    return onnx_available_devices + hailo_available_devices


def hailort_avilable() -> bool:
    """
    Checks if Hailo RT is available on the system.

    Returns True if Hailo devices are present and the Hailo SDK is installed, False otherwise.
    """
    if get_hailo_devices():
        try:
            import hailo_platform

            return True
        except ImportError:
            return False
    return False


def get_optimal_providen() -> str:
    """
    Retrieves the optimal provider for performing inference.

    The function checks if Hailo devices are present and the Hailo SDK is installed. If so, it returns the first
    available Hailo device. Otherwise, it returns the first available provider from the list of available ONNX providers.

    Returns:
        str: The optimal provider for inference.
    """
    hailo_devices = get_hailo_devices()
    if hailo_devices and hailort_avilable():
        return hailo_devices[0]
    onnx_available_devices = C.get_available_providers()
    return onnx_available_devices[0]


def check_hailo_device() -> bool:
    """
    Checks if a Hailo device is present in the system.

    Returns:
        bool: True if a Hailo device is present, False otherwise.
    """
    output = subprocess.run("lspci", capture_output=True, check=True).stdout.decode()

    return "Hailo" in output


if __name__ == "__main__":
    print(check_hailo_device())
