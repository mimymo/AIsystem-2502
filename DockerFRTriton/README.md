# DockerFRTriton (HW2)

A Face Recognition API served by Triton Inference Server (CPU) with a FastAPI wrapper.

## Overview

* Triton serves ONNX models from `model_repository/`
* FastAPI exposes the following endpoints:

  * `/embedding`
  * `/face-similarity`
  * `/health`
* All neural network inference is executed exclusively on Triton
* Anti-spoofing is NOT used (per instructor note)


## Requirements

* Docker Desktop (Windows / Mac / Linux)
* Available ports: 3000, 8000, 8001, 8002


## Model Repository Structure

```text
model_repository/
└── fr_model/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

* `fr_model` is an ArcFace-based face recognition backbone exported to ONNX
* The model is served on CPU using the Triton ONNX Runtime backend


## API Endpoints

### `POST /embedding`

Returns a 512-dimensional face embedding for a single input image.

### `POST /face-similarity`

Returns a similarity score in the range [0, 1] for two input face images.

### `GET /health`

Health check endpoint for the API server.


## Similarity Calibration (Important)

The face recognition model outputs L2-normalized embeddings.
Similarity is computed using cosine similarity between embeddings.

In practice, raw cosine similarity scores between different identities
(particularly same-gender pairs) can be relatively high when using a
lightweight ArcFace model on unconstrained images.

To achieve clearer separation between same-person and different-person pairs,
a lightweight score calibration layer is applied on top of the raw cosine similarity.

* This calibration:

  * Preserves ranking
  * Improves decision margins
  * Reflects common practice in real-world face recognition systems

Note:
All neural network inference (embedding extraction) is still performed
entirely by Triton Inference Server.


## Run with Docker

### Build the image

```bash
docker build -t fr-triton -f Docker/Dockerfile .
```

### Run the container

```bash
docker run --rm \
  -p 3000:3000 \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --name fr_triton \
  fr-triton
```


## API Documentation (Swagger)

After the container is running, access:

```
http://localhost:3000/docs
```


## Notes

* Triton runs on CPU only (GPU is not required)
* FastAPI acts as a thin wrapper and does not execute any model locally
* Calibration logic is applied only at the similarity score level
