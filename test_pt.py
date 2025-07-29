# test_pt.py
import torch
from PIL import Image
import torchvision.transforms as T
import argparse
import os
from aodnet_model import AODNet

def load_image(img_path, img_size):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1, 3, H, W]

def save_output(tensor, out_path):
    tensor = tensor.squeeze().clamp(0, 1).cpu()
    to_pil = T.ToPILImage()
    img = to_pil(tensor)
    img.save(out_path)

def main(args):
    device = torch.device(args.device)

    # Load model
    model = AODNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load image
    input_tensor = load_image(args.input, args.img_size).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dehazed_pt.png")
    save_output(output, out_path)
    print(f"[PyTorch] Output saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--input", type=str, required=True, help="Input hazy image")
    parser.add_argument("--output_dir", type=str, default="results_pt")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
