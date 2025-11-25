# DockerFRTriton

Skeleton homework for serving a face-recognition (FR) model on Triton Inference Server. The FR model should be converted to ONNX, run on CPU, and exposed via port forwarding from the container.

## What to implement

- Convert your FR model to ONNX (CPU-friendly) in `convert_to_onnx.py`.
- Stand up a minimal Triton server that loads the ONNX model and serves it in `triton_service.py`.
- Expose Triton HTTP/GRPC/metrics ports (e.g., `8000/8001/8002`) with Docker port forwarding.

All functions are stubs; fill them in for the assignment.

## Quickstart (expected flow)

1) Install deps (optional for local testing)
```bash
cd DockerFRTriton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Convert to ONNX (fill in logic first)
```bash
python convert_to_onnx.py \
  --weights-path weights/your_fr_model.pth \
  --onnx-path model_repository/fr_model/1/model.onnx \
  --opset 12
```

3) Build & run the Docker image with Triton (after implementing)
```bash
docker build -t fr-triton -f Docker/Dockerfile .
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 fr-triton
```

You can then query Triton at `http://localhost:8000/v2/health/ready` or via a client once you wire up the code.
