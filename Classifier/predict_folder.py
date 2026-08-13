import argparse
import os
import csv
from PIL import Image

import torch
import torch.nn as nn
import open_clip


class CLIPBinaryClassifier(nn.Module):

    def __init__(self, clip_model):
        super().__init__()

        self.clip = clip_model

        embed_dim = clip_model.visual.output_dim

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):

        feat = self.clip.encode_image(x)

        feat = feat / feat.norm(
            dim=-1,
            keepdim=True
        )

        logits = self.head(feat)

        return logits.squeeze(1)


def load_model(weights_path, device):

    print("Loading CLIP...")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai"
    )

    for p in clip_model.parameters():
        p.requires_grad = False

    for p in clip_model.visual.transformer.resblocks[-1].parameters():
        p.requires_grad = True

    for p in clip_model.visual.transformer.resblocks[-2].parameters():
        p.requires_grad = True

    model = CLIPBinaryClassifier(
        clip_model
    ).to(device)

    print("Loading weights...")

    model.load_state_dict(
        torch.load(
            weights_path,
            map_location=device
        )
    )

    model.eval()

    return model, preprocess


def predict(model, preprocess, image_path, device):

    img = Image.open(image_path).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

        prob_fake = torch.sigmoid(logits).item()

    prob_real = 1.0 - prob_fake

    prediction = "FAKE" if prob_fake > 0.5 else "REAL"

    return prediction, prob_real, prob_fake


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing SEM images"
    )

    parser.add_argument(
        "--weights",
        default="clip_finetuned_nanoparticle.pt",
        help="Model checkpoint"
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="CSV output file"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = load_model(
        args.weights,
        device
    )

    valid_ext = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    )

    image_files = sorted([
        f for f in os.listdir(args.folder)
        if f.lower().endswith(valid_ext)
    ])

    print(f"\nFound {len(image_files)} images.\n")

    with open(args.output, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "Image",
            "Prediction",
            "Real_Probability",
            "Fake_Probability"
        ])

        for fname in image_files:

            image_path = os.path.join(
                args.folder,
                fname
            )

            prediction, prob_real, prob_fake = predict(
                model,
                preprocess,
                image_path,
                device
            )

            writer.writerow([
                fname,
                prediction,
                f"{prob_real:.6f}",
                f"{prob_fake:.6f}"
            ])

            print(
                f"{fname:30s} "
                f"{prediction:5s} "
                f"Real={prob_real:.4f} "
                f"Fake={prob_fake:.4f}"
            )

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()