# Placeholder Triton server image for CPU serving of the FR ONNX model.
# Fill in with the correct Triton base image and startup command once your model is ready.

FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Copy your prepared model repository (update path if different)
COPY model_repository /models

# Default ports for Triton: HTTP 8000, gRPC 8001, metrics 8002
EXPOSE 8000 8001 8002

# TODO: Replace with the actual launch command once you wire up the repo/configs.
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
