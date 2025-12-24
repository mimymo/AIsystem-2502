# DockerFRTriton (HW2)

Face Recognition API served by **Triton Inference Server (CPU)** with a **FastAPI** wrapper.

This folder (`DockerFRTriton/`) is the **submission root** for HW2.

* Triton serves ONNX models from `model_repository/`
* FastAPI exposes `/embedding`, `/face-similarity`, `/health`
* All neural network inference is executed **exclusively on Triton**
* Anti-spoofing is **not used** (per instructor note)



## Requirements

* Docker Desktop (Windows / Mac / Linux)
* Available ports: **3000**, **8000**, **8001**, **8002**



## Folder Structure

```
DockerFRTriton/
├── Docker/
│   ├── Dockerfile
│   └── start.sh
├── model_repository/
│   └── fr_model/
│       ├── 1/
│       │   └── model.onnx
│       └── config.pbtxt
├── app.py
├── pipeline.py
├── triton_service.py
├── convert_to_onnx.py
├── run_fastapi.py
├── requirements.txt
└── README.md
```

> **Note:**
> All paths in this README are relative to the `DockerFRTriton/` directory.



## Model Repository

* `fr_model` is an **ArcFace-based face recognition backbone** exported to ONNX
* The model is served on **CPU** using Triton **ONNX Runtime backend**
* Embedding dimension: **512**



## API Endpoints

### `POST /embedding`

Returns a **512-D face embedding** for a single input image.

### `POST /face-similarity`

Returns a **similarity score in [0, 1]** for two input face images.

### `GET /health`

Health check endpoint.



## Similarity Computation

* The FR model outputs **L2-normalized embeddings**
* Raw **cosine similarity** is computed between embeddings
* A **lightweight calibration layer** is applied to improve separation
  between same-person and different-person pairs

All neural network inference (embedding extraction) is still performed
**entirely by Triton Inference Server**.



## Run with Docker

From inside the `DockerFRTriton/` directory:

```bash
docker build -t fr-triton -f Docker/Dockerfile .
docker run --rm \
  -p 3000:3000 \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --name fr_triton \
  fr-triton
```



## Swagger UI

```
http://localhost:3000/docs
```



## Notes

* Triton runs on **CPU only** (GPU not required)
* FastAPI acts as a **thin wrapper** and does not run models locally
* All model inference is handled by Triton
* Calibration logic is applied only at the **similarity score level**

