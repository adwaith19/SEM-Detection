import argparse
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

    clip_model, _, preprocess = \
        open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai"
        )

    #
    # Same architecture used during training
    #

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


def predict(model,
            preprocess,
            image_path,
            device):

    img = Image.open(
        image_path
    ).convert("RGB")

    x = preprocess(
        img
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

        prob_fake = torch.sigmoid(
            logits
        ).item()

    prob_real = 1.0 - prob_fake

    prediction = (
        "FAKE"
        if prob_fake > 0.5
        else "REAL"
    )

    return (
        prediction,
        prob_real,
        prob_fake
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="SEM image"
    )

    parser.add_argument(
        "--weights",
        default="clip_finetuned_nanoparticle.pt",
        help="Model checkpoint"
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, preprocess = load_model(
        args.weights,
        device
    )

    prediction, prob_real, prob_fake = predict(
        model,
        preprocess,
        args.image,
        device
    )

    print()

    print("Prediction")
    print("------------------------")

    print(
        f"Real probability : {prob_real:.4f}"
    )

    print(
        f"Fake probability : {prob_fake:.4f}"
    )

    print(
        f"Predicted class  : {prediction}"
    )


if __name__ == "__main__":

    main()
