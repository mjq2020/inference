"""Native, single-process server entry point."""

import argparse

import uvicorn

from .api import create_app
from .settings import EdgeSettings


def main():
    parser = argparse.ArgumentParser(description="RKNN-only Inference runtime for RV1126B")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-root")
    args = parser.parse_args()
    settings = EdgeSettings.from_env()
    from dataclasses import replace
    from pathlib import Path

    if args.host is not None:
        settings = replace(settings, host=args.host)
    if args.port is not None:
        settings = replace(settings, port=args.port)
    if args.model_root is not None:
        settings = replace(settings, model_root=Path(args.model_root))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, access_log=False)
