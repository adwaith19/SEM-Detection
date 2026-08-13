import os
import argparse
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score
)
import random
import numpy as np
import open_clip

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def seed_everything(seed=456):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#torch.use_deterministic_algorithms(True)

def collect(root):

    items = []

    for dirpath, _, files in os.walk(root):

        low = dirpath.replace('\\', '/').lower()

        if '/0_real' in low or low.endswith('0_real'):
            label = 0

        elif '/1_fake' in low or low.endswith('1_fake'):
            label = 1

        else:
            continue

        for f in files:

            if f.lower().endswith(IMG_EXT):

                items.append(
                    (
                        os.path.join(dirpath, f),
                        label
                    )
                )

    return items


class ImgSet(Dataset):

    def __init__(self, items, preprocess):

        self.items = items
        self.pre = preprocess

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):

        path, label = self.items[idx]

        img = Image.open(path).convert("RGB")

        img = self.pre(img)

        return img, label


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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.float().to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict_dataset(
    model,
    loader,
    device
):

    model.eval()

    probs = []
    labels = []

    for x, y in loader:

        x = x.to(device)

        logits = model(x)

        p = torch.sigmoid(logits)

        probs.extend(
            p.cpu().numpy()
        )

        labels.extend(
            y.numpy()
        )

    return (
        labels,
        probs
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_dir',
        default='./dataset/train'
    )

    parser.add_argument(
        '--test_root',
        default='./dataset/test'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=15
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=32
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4
    )

    args = parser.parse_args()
    
    seed_everything(456) ###

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print("Loading CLIP...")

    clip_model, _, preprocess = \
        open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='openai'
        )

    #
    # Freeze everything
    #

    for p in clip_model.parameters():

        p.requires_grad = False

    #
    # Unfreeze last two transformer blocks
    #

    for p in clip_model.visual.transformer.resblocks[-1].parameters():

        p.requires_grad = True

    for p in clip_model.visual.transformer.resblocks[-2].parameters():

        p.requires_grad = True

    model = CLIPBinaryClassifier(
        clip_model
    ).to(device)

    print("Preparing datasets...")

    train_items = collect(
        args.train_dir
    )

    train_loader = DataLoader(
        ImgSet(train_items, preprocess),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-5,
        weight_decay=1e-4
    )

    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining")

    for epoch in range(args.epochs):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} "
            f"Loss={loss:.4f}"
        )

    #
    # Save model
    #

    torch.save(
        model.state_dict(),
        "clip_finetuned_nanoparticle.pt" ###
    )

    print(
        "\nSaved model -> clip_finetuned_sem.pt"
    )

    #
    # Evaluate
    #

    cats = sorted([
        d
        for d in os.listdir(args.test_root)
        if os.path.isdir(
            os.path.join(
                args.test_root,
                d
            )
        )
    ])

    print(
        '\n{:<18} {:>8} {:>8}'.format(
            'testset',
            'acc',
            'ap',
	    'RealAcc',
            'FakeAcc'
        )
    )

    for c in cats:

        items = collect(
            os.path.join(
                args.test_root,
                c
            )
        )

        if not items:
            continue

        loader = DataLoader(
            ImgSet(
                items,
                preprocess
            ),
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.workers
        )

        y_true, probs = predict_dataset(
            model,
            loader,
            device
        )

        acc = accuracy_score(
            y_true,
            np.array(probs) > 0.5
        )

        ap = average_precision_score(
            y_true,
            probs
        )

        preds = (np.array(probs) > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            preds,
            labels=[0, 1]
        ).ravel()

        real_acc = tn / (tn + fp)
        fake_acc = tp / (tp + fn)

       # print(
       #     '{:<18} {:>8.3f} {:>8.3f}'.format(
       #         c,
       #         acc,
       #         ap
       #     )
       # )
        print(
            '{:<18} Acc={:.3f}  AP={:.3f}  Real={:.3f}  Fake={:.3f}'.format(
                c,
                acc,
                ap,
                real_acc,
                fake_acc
            )
        )
        
    all_items = collect(args.test_root)
        
    loader = DataLoader(
        ImgSet(
            all_items,
            preprocess
        ),
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers
    )  

    y_true, probs = predict_dataset(
        model,
        loader,
        device
    ) 

    overall_acc = accuracy_score(
        y_true,
        np.array(probs) > 0.5
    )

    overall_ap = average_precision_score(
        y_true,
        probs
    )

    print("\nOverall Test Performance")
    print("------------------------")
    print(f"Accuracy : {overall_acc:.3f}")
    print(f"AP       : {overall_ap:.3f}")

if __name__ == '__main__':
    import numpy as np
    main()
