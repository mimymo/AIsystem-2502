from typing import Any, Tuple
import numpy as np

from triton_service import run_inference


def _l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def get_embeddings(
    client: Any,
    image_a: bytes,
    image_b: bytes
) -> Tuple[np.ndarray, np.ndarray]:
    emb_a = run_inference(client, image_a).squeeze(0).astype(np.float32)
    emb_b = run_inference(client, image_b).squeeze(0).astype(np.float32)

    emb_a = _l2_normalize(emb_a)
    emb_b = _l2_normalize(emb_b)

    return emb_a, emb_b


def calculate_face_similarity(
    client: Any,
    image_a: bytes,
    image_b: bytes
) -> float:
    emb_a, emb_b = get_embeddings(client, image_a, image_b)

    raw = float(np.sum(emb_a * emb_b))

   low = 0.55
    high = 0.75

    if raw <= low:
        score = (raw / low) * 0.30

    elif raw >= high:
        score = 0.60 + (raw - high) * 1.5

    else:
        t = (raw - low) / (high - low)  # 0..1
        score = 0.30 + (t ** 4) * 0.30  # 強く圧縮

    return float(np.clip(score, 0.0, 1.0))
