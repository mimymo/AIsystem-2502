from pathlib import Path
from typing import Any


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002


def prepare_model_repository(model_repo: Path) -> None:
    """
    TODO: Populate the Triton model repository with your ONNX FR model and config.pbtxt.
    """
    pass


def start_triton_server(model_repo: Path) -> Any:
    """
    TODO: Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    pass


def stop_triton_server(server_handle: Any) -> None:
    """
    TODO: Cleanly stop the Triton server started in start_triton_server.
    """
    pass


def create_triton_client(url: str) -> Any:
    """
    TODO: Initialize a Triton HTTP client for the FR model endpoint.
    """
    pass


def run_inference(client: Any, image_bytes: bytes) -> Any:
    """
    TODO: Preprocess an input image, call Triton, and decode embeddings or scores.
    """
    pass
