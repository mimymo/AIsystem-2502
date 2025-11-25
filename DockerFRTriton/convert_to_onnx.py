import argparse
from pathlib import Path


def convert_model_to_onnx(weights_path: Path, onnx_path: Path, opset: int) -> None:
    """
    TODO: Load your FR model weights, trace it on CPU, and export to ONNX.
    """
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FR model to ONNX for Triton.")
    parser.add_argument("--weights-path", type=Path, required=True, help="Path to FR model weights (.pth).")
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/fr_model/1/model.onnx"),
        help="Destination for exported ONNX file.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_model_to_onnx(args.weights_path, args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
