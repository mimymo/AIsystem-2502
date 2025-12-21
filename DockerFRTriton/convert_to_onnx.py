import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ArcFaceResNet18(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def convert_to_onnx(onnx_path: Path, opset: int):
    model = ArcFaceResNet18()
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 112, 112)
    dynamic_axes = {
        "input": {0: "batch"},
        "embedding": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[OK] ArcFace FR ONNX exported to {onnx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/fr_model/1/model.onnx"),
    )
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    convert_to_onnx(args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
