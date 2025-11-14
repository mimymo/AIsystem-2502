"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List

import numpy as np
import cv2

try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    raise ImportError(
        "insightface is required for this implementation. "
        "Please install it with `pip install insightface`."
    ) from e




_face_app: FaceAnalysis | None = None


def _get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
    return _face_app


def _decode_image(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, (bytes, bytearray)):
        data = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
        return img

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Expected raw bytes, bytearray, or numpy.ndarray."
    )


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    v1 = vec1.astype(np.float32)
    v2 = vec2.astype(np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    cos = float(np.dot(v1, v2) / (norm1 * norm2))
    sim = (cos + 1.0) / 2.0
    return max(0.0, min(1.0, sim))




def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.

    Parameters can be raw image bytes or a decoded image object, depending on
    the student implementation. Expected to return a list of face regions
    (e.g., bounding boxes or cropped images).
    """

    bgr = _decode_image(image)
    app = _get_face_app()
    faces = app.get(bgr)
    return list(faces)


def compute_face_embedding(face_image: Any) -> np.ndarray:
    """
    Compute a numerical embedding vector for the provided face image.

    The embedding should capture discriminative facial features for comparison.
    """
    if hasattr(face_image, "embedding"):
        emb = np.asarray(face_image.embedding, dtype=np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        return emb

    bgr = _decode_image(face_image)
    app = _get_face_app()
    faces = app.get(bgr)
    if not faces:
        raise ValueError("No face detected when computing embedding.")

    best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    emb = np.asarray(best_face.embedding, dtype=np.float32)
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    return emb


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.

    The return type can be tailored to the chosen keypoint detection library.
    """
    if hasattr(face_image, "kps"):
        return np.asarray(face_image.kps, dtype=np.float32)

    bgr = _decode_image(face_image)
    app = _get_face_app()
    faces = app.get(bgr)
    if not faces:
        raise ValueError("No face detected for keypoint detection.")

    best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    return np.asarray(best_face.kps, dtype=np.float32)


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.

    Typically used to align faces prior to embedding extraction.
    """
    bgr = _decode_image(image)
    H = np.asarray(homography_matrix, dtype=np.float32)
    h, w = bgr.shape[:2]
    warped = cv2.warpPerspective(bgr, H, (w, h))
    return warped


def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    bgr = _decode_image(face_image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()

    score = var / 1000.0
    score = max(0.0, min(1.0, float(score)))
    return score


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline that returns a similarity score between two faces.

    This function should:
      1. Detect faces in both images.
      2. Align faces using keypoints and homography warping.
      3. (Run anti-spoofing checks to validate face authenticity. - If you want)
      4. Generate embeddings and compute a similarity score.

    The images provided by the API arrive as raw byte strings; convert or decode
    them as needed for downstream processing.
    """
    bgr_a = _decode_image(image_a)
    bgr_b = _decode_image(image_b)

    faces_a = detect_faces(bgr_a)
    faces_b = detect_faces(bgr_b)

    if not faces_a or not faces_b:
        return 0.0

    face_a = max(faces_a, key=lambda f: float(getattr(f, "det_score", 0.0)))
    face_b = max(faces_b, key=lambda f: float(getattr(f, "det_score", 0.0)))

    emb_a = compute_face_embedding(face_a)
    emb_b = compute_face_embedding(face_b)

    similarity = _cosine_similarity(emb_a, emb_b)
    return similarity
