import os
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer

# ------------------ Accuracy Top-1/Top-5 ------------------ #
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ------------------ Contrastive Loss ------------------ #
def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.matmul(z, z.T) / tau

    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)  # avoid self-similarity

    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / tau)
    pos = torch.cat([pos, pos], dim=0)

    denom = torch.exp(sim).sum(dim=1)
    loss = -torch.log(pos / denom)
    return loss.mean()


# ------------------ SimCLR Lightning Module ------------------ #
class SimCLR(pl.LightningModule):
    def __init__(self, lr=1e-3, tau=0.5):
        super().__init__()
        self.save_hyperparameters()
        base_encoder = torchvision.models.resnet18(pretrained=False)
        base_encoder.fc = nn.Identity()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.tau = tau
        self.lr = lr

    def forward(self, x):
        features = self.encoder(x)
        return self.projection_head(features)

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = contrastive_loss(z1, z2, self.tau)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ------------------ Linear Evaluation ------------------ #
class LinearEval(pl.LightningModule):
    def __init__(self, encoder, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        self.lr = lr

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)
        return self.classifier(feat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc1, acc5 = accuracy(logits, y)
        self.log("train/acc1", acc1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/acc5", acc5, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc1, acc5 = accuracy(logits, y)
        self.log("val/acc1", acc1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val/acc5", acc5, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)


# ------------------ Data ------------------ #
from torchvision import transforms
import numpy as np
import torchvision.datasets as datasets
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class CIFAR10Pairs(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        img2, _ = super().__getitem__(index)
        return (img, img2), 0


def get_simclr_transforms():
    return T.Compose([
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * 32)),
        T.ToTensor()
    ])


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
def main(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_dir = os.path.join("tb_logs", "simclr")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    existing_versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    num_version = len(existing_versions) + 1

    version_name = f"v{num_version}_eval_{args.frac}"
    tb_logger = TensorBoardLogger("tb_logs", name="simclr", version=version_name)

    csv_logger = CSVLogger("csv_logs", name="simclr")
    # Logger setup
    if not args.only_eval:
        log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.info("Starting training...")

        # Checkpoint + Early stopping
        ckpt_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="simclr-{epoch:02d}-{train_loss:.4f}",
            monitor="train/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        early_stop = EarlyStopping(
            monitor="train/loss",
            mode="min",
            patience=10,
            verbose=True
        )

        # Prepare data
        transform = get_simclr_transforms()
        train_ds = CIFAR10Pairs(root="./data", train=True, transform=transform, download=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        model = SimCLR(lr=args.lr, tau=args.tau)

        # Resume if required
        resume_path = "checkpoints/simclr/epoch=99-step=19600.ckpt" if args.resume else None
        if resume_path and os.path.exists(resume_path):
            logging.info(f"Resuming from checkpoint: {resume_path}")

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            logger=[tb_logger, csv_logger],
            callbacks=[ckpt_callback, early_stop],
            log_every_n_steps=10,
            resume_from_checkpoint=resume_path if args.resume else None,
        )
        trainer.fit(model, train_loader)

        torch.save(model.encoder.state_dict(), "encoder.pth")

    logging.info("Starting linear evaluation...")

    # Eval transform
    eval_transform = T.Compose([T.ToTensor()])

    # Load full train set
    full_train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, transform=eval_transform)
    full_targets = full_train_ds.targets

    # Stratified split to keep label distribution
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.frac, random_state=42)
    indices, _ = next(splitter.split(np.zeros(len(full_targets)), full_targets))

    # Create subset
    subset_train_ds = torch.utils.data.Subset(full_train_ds, indices)

    print("Fraction: ", args.frac)
    print("Length of full targets: ", len(full_targets))
    print("Length of subset: ", subset_train_ds.__len__())

    val_ds = torchvision.datasets.CIFAR10(root="./data", train=False, transform=eval_transform)

    train_loader = DataLoader(subset_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load encoder and linear eval model
    encoder = torchvision.models.resnet18(pretrained=False)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load("encoder.pth"))
    linear_model = LinearEval(encoder)

    early_stop_eval = EarlyStopping(
        monitor="val/acc1",
        mode="max",          
        patience=10,
        verbose=True
    )

    ckpt_eval = ModelCheckpoint(
        dirpath="checkpoints",
        filename="linear-eval-{epoch:02d}-{val_acc1:.2f}",
        monitor="val/acc1",
        mode="max",
        save_top_k=1
    )

    eval_trainer = Trainer(
        max_epochs=args.eval_epochs,
        accelerator="gpu",
        logger=[tb_logger, csv_logger],
        callbacks=[early_stop_eval, ckpt_eval]
    )

    eval_trainer.fit(linear_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--eval_epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--only_eval", action="store_true", help="Only evaluate the model")
    parser.add_argument("--frac", type=float, default=0.1, help="Fraction of labels to use for training (must be >0 and <1)")
    if not (0 < parser.parse_args().frac < 1):
        raise ValueError("The --frac argument must be >0 and <1")
    args = parser.parse_args()

    main(args)