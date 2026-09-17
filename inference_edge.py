"""Set the device profile before Python imports the inference package."""

import os


def main():
    os.environ["INFERENCE_RUNTIME_PROFILE"] = "rv1126b"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from inference.edge.server import main as serve

    serve()


if __name__ == "__main__":
    main()
