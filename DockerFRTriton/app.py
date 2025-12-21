import logging
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from pipeline import calculate_face_similarity
from triton_service import (
    TRITON_HTTP_PORT,
    create_triton_client,
    run_inference,
)

app = FastAPI(
    title="FR Triton API",
    description="Minimal FastAPI wrapper around Triton Inference Server for FR embeddings.",
    version="0.1.0",
)

_triton_client: Optional[Any] = None
logger = logging.getLogger("fr_triton_app")


@app.on_event("startup")
def startup_event() -> None:
    """
    Connect to Triton server started by Docker (start.sh).
    """
    global _triton_client
    try:
        _triton_client = create_triton_client(f"localhost:{TRITON_HTTP_PORT}")
        logger.info("Connected to Triton server.")
    except Exception as exc:
        logger.exception("Failed to connect to Triton: %s", exc)
        _triton_client = None


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embedding", tags=["Face Recognition"])
async def embedding(
    image: UploadFile = File(..., description="Face image to embed")
) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    content = await image.read()
    embedding_arr = run_inference(_triton_client, content)
    embedding_list = embedding_arr.reshape(embedding_arr.shape[0], -1).tolist()
    return {"embedding": embedding_list}


@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
) -> dict[str, Any]:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client is not initialized.")

    img_a = await image_a.read()
    img_b = await image_b.read()
    score = calculate_face_similarity(_triton_client, img_a, img_b)
    return {"similarity": score}
