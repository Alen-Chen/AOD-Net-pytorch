import torch
import argparse
import os
from aodnet_model import AODNet


def export_onnx(model_path, output_path, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AODNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=11
    )

    print(f"ONNX model exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AOD-Net to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--output_path", type=str, default="aodnet.onnx", help="Path to save ONNX model")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for dummy input (HxW)")

    args = parser.parse_args()
    export_onnx(args.model_path, args.output_path, args.img_size)
