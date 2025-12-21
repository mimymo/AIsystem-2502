import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from io import BytesIO

from tritonclient import http as httpclient


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input"
MODEL_OUTPUT_NAME = "embedding"
MODEL_IMAGE_SIZE = (112, 112)



def start_triton_server(model_repo: Path) -> Any:
    """
    Start Triton Inference Server (CPU mode).
    """
    cmd = [
        "tritonserver",
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--log-verbose=0",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # wait for Triton to boot
    time.sleep(5)
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Stop Triton server safely.
    """
    if server_handle is None:
        return
    try:
        server_handle.terminate()
    except Exception:
        pass


def create_triton_client(url: str):
    """
    Create Triton HTTP client.
    """
    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def run_inference(client: Any, image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image and run inference on Triton.
    Returns embedding: (1, 512)
    """
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = _center_crop_square(img)
        img = img.resize(MODEL_IMAGE_SIZE)

        x = np.asarray(img, dtype=np.float32)

    x = (x - 127.5) / 128.0

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    infer_input = httpclient.InferInput(
        MODEL_INPUT_NAME,
        x.shape,
        "FP32",
    )
    infer_input.set_data_from_numpy(x)

    infer_output = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )

    return response.as_numpy(MODEL_OUTPUT_NAME)
