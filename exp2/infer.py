"""Use a trained AlexNet checkpoint to classify one external image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.alexnet import AlexNet


CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010],
    ),
])


def load_checkpoint(path, map_location="cpu"):
    """Load checkpoints on both older and newer PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_best_checkpoint(weights_dir):
    """Return the checkpoint with the largest saved test accuracy."""
    candidates = sorted(weights_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in: {weights_dir}")

    valid = []
    for path in candidates:
        try:
            checkpoint = load_checkpoint(path)
            accuracy = float(checkpoint["acc"])
            epoch = int(checkpoint["epoch"])
            if "net" not in checkpoint:
                raise KeyError("net")
            valid.append((accuracy, epoch, path, checkpoint))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            print(f"Skipping invalid checkpoint {path.name}: {error}")

    if not valid:
        raise RuntimeError(f"No valid AlexNet checkpoint found in: {weights_dir}")

    # Accuracy is the main criterion; a later epoch breaks an exact tie.
    return max(valid, key=lambda item: (item[0], item[1]))


def choose_font(size=30):
    """Use a common TrueType font, with Pillow's default as a fallback."""
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_prediction(image, text):
    """Draw readable prediction text without changing the inference input."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = choose_font()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    padding = 8
    box = (0, 0, right - left + padding * 2, bottom - top + padding * 2)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((padding, padding - top), text, font=font, fill=(255, 255, 0))
    return result


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="AlexNet single-image inference")
    parser.add_argument("--image", type=Path, default=script_dir / "test.jpeg")
    parser.add_argument("--output", type=Path, default=script_dir / "test_result.jpeg")
    parser.add_argument(
        "--weights-dir", type=Path, default=script_dir / "weights" / "alexnet"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    accuracy, epoch, checkpoint_path, checkpoint = find_best_checkpoint(args.weights_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device)
    model.load_state_dict(checkpoint["net"])
    model.eval()

    # convert("RGB") also handles grayscale/RGBA internet images safely.
    with Image.open(args.image) as opened_image:
        image = opened_image.convert("RGB")

    input_tensor = PREPROCESS(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    confidence, class_index = probabilities.max(dim=0)
    label = CLASSES[class_index.item()]
    prediction_text = f"{label} ({confidence.item() * 100:.2f}%)"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_prediction(image, prediction_text).save(args.output, quality=95)

    print(f"Device: {device}")
    print(f"Selected checkpoint: {checkpoint_path}")
    print(f"Checkpoint test accuracy: {accuracy:.3f}% (epoch {epoch + 1})")
    print(f"Prediction: {prediction_text}")
    print(f"Result image: {args.output}")


if __name__ == "__main__":
    main()
