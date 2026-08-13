import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import CLIPBinaryClassifier


device = "cpu"

# Load model
model = CLIPBinaryClassifier()

state = torch.load(
    "../clip_finetuned_biological.pt",
    map_location=device
)

model.load_state_dict(state)
model.eval()

preprocess = model.preprocess


# reshape ViT tokens
def reshape_transform(tensor):

    tensor = tensor[:, 1:, :]        # remove CLS token

    tensor = tensor.reshape(
        tensor.size(0),
        16,
        16,
        tensor.size(2)
    )

    tensor = tensor.permute(
        0,
        3,
        1,
        2
    )

    return tensor


# Target layer
target_layers = [
    model.clip.visual.transformer.resblocks[-1]
]


# CAM wrapper
class CAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y.unsqueeze(1)


cam_model = CAMWrapper(model)

cam = EigenCAM(
    model=cam_model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

targets = [ClassifierOutputTarget(0)]


# Folder containing images
image_folder = "path/to/folder/containing/images/" ###


valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


for filename in sorted(os.listdir(image_folder)):

    if not filename.lower().endswith(valid_ext):
        continue

    # Skip already generated overlays
    if filename.endswith("_overlay.png"):
        continue

    img_path = os.path.join(image_folder, filename)

    pil = Image.open(img_path).convert("RGB")

    input_tensor = preprocess(pil).unsqueeze(0)

    rgb_img = np.array(pil).astype(np.float32) / 255.0


    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]


    with torch.no_grad():

        logit = model(input_tensor)

        prob = torch.sigmoid(logit).item()

    prediction = "Fake" if prob > 0.5 else "Real"

    print(f"{filename:25s} -> {prediction:5s} ({prob:.4f})")



    rgb_img_resized = cv2.resize(rgb_img, (224, 224))

    visualization = show_cam_on_image(
        rgb_img_resized,
        grayscale_cam,
        use_rgb=True
    )

    base = os.path.splitext(filename)[0]

    output_path = os.path.join(
        image_folder,
        f"{base}_overlay.png"
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

print("Done!")
