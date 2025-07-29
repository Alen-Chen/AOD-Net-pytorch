# test_onnx.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
import os
import torchvision.transforms as T

def load_image(img_path, img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor.numpy()

def save_output(output_np, out_path):
    output_np = np.clip(output_np.squeeze(), 0, 1)
    img = Image.fromarray((output_np.transpose(1, 2, 0) * 255).astype(np.uint8))
    img.save(out_path)

def main(args):
    # Load ONNX session
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Load image
    input_tensor = load_image(args.input, args.img_size).astype(np.float32)

    # Inference
    output = sess.run([output_name], {input_name: input_tensor})[0]

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dehazed_onnx.png")
    save_output(output, out_path)
    print(f"[ONNX] Output saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--input", type=str, required=True, help="Input hazy image")
    parser.add_argument("--output_dir", type=str, default="results_onnx")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
