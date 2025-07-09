#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_robust.py
Train a robust classifier against FGSM/PGD attacks and export the model for submission.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ---------- Dataset ----------
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        return self.ids[index], img, self.labels[index]

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


sys.modules['__main__'].MembershipDataset = MembershipDataset

# ---------- Argument Parser ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="Train.pt", help="Training set path (.pt)")
    p.add_argument("--out", type=str, default="robust_model.pt", help="Output filename (.pt)")
    p.add_argument("--model", type=str, default="resnet34",
                   choices=["resnet18", "resnet34", "resnet50"], help="Model architecture")
    p.add_argument("--token", type=str, default="34811541", help="Token assigned for evaluation")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eps", type=float, default=8/255, help="FGSM/PGD epsilon")
    p.add_argument("--alpha", type=float, default=2/255, help="PGD step size or FGSM step")
    p.add_argument("--pgd_steps", type=int, default=7, help="Number of PGD iterations")
    p.add_argument("--method", choices=["pgd", "fast"], default="pgd",
                   help="pgd = Madry PGD-AT | fast = Fast FGSM-AT")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ---------- Data Loading ----------
def get_loaders(dataset_path, batch_size):
    dataset = torch.load(dataset_path, weights_only=False)
    print(f"Dataset length: {len(dataset)}")
    print("Example sample:", dataset[0])

    _, img, _, *_ = dataset[0]
    if isinstance(img, Image.Image):
        print("Image is PIL, converting to tensor.")
    elif isinstance(img, torch.Tensor):
        print("Image is tensor:", img.shape)
    else:
        print("Unknown image type:", type(img))

    print("Image size:", np.array(img).shape if isinstance(img, np.ndarray) else img.size)
    input()

    mean = [0.2980, 0.2962, 0.2987]
    std = [0.2886, 0.2875, 0.2889]

    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset.transform = train_tf
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)

    val_size = max(1, len(dataset) // 20)
    indices = np.random.choice(len(dataset), size=val_size, replace=False)
    val_ds = torch.utils.data.Subset(dataset, indices)
    val_ds.dataset.transform = test_tf
    val_loader = DataLoader(val_ds, batch_size=256,
                            shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader

# ---------- Model ----------
def get_model(name):
    assert name in ["resnet18", "resnet34", "resnet50"]
    model_ctor = getattr(models, name)
    net = model_ctor(weights=None, num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net

# ---------- Adversarial Attack ----------
def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    eta = eps * x_adv.grad.data.sign()
    return (x_adv + eta).clamp(0, 1).detach()

def pgd(model, x, y, eps, alpha, steps):
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            eta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = (x + eta).clamp(0, 1).detach()
    return x_adv

# ---------- Evaluation ----------
def eval_acc(model, loader, device, attack=None, **atk_kwargs):
    model.eval()
    correct, total = 0, 0
    for _, imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if attack is not None:
            imgs = attack(model, imgs, labels, **atk_kwargs)
        logits = model(imgs)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total

# ---------- Training ----------
warm_up = 3

def train(args):
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    device = torch.device("mps")
    print("Device:", device)

    train_loader, val_loader = get_loaders(args.data, args.batch_size)
    model = get_model(args.model).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}", leave=False)
        for _, imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            if ep < warm_up:
                imgs_adv = imgs
            elif args.method == "pgd":
                imgs_adv = pgd(model, imgs, labels, eps=args.eps,
                               alpha=args.alpha, steps=args.pgd_steps)
            else:
                imgs_adv = fgsm(model, imgs, labels, eps=args.alpha)

            logits_clean = model(imgs)
            logits_adv = model(imgs_adv)

            loss_clean = F.cross_entropy(logits_clean, labels)
            loss_adv = F.cross_entropy(logits_adv, labels)
            loss = 0.5 * loss_clean + 0.5 * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        clean_acc = eval_acc(model, val_loader, device)
        fgsm_acc = eval_acc(model, val_loader, device, attack=fgsm, eps=args.eps)
        pgd_acc = eval_acc(model, val_loader, device, attack=pgd,
                           eps=args.eps, alpha=args.alpha, steps=args.pgd_steps)
        print(f"[Epoch {ep:02d}] Clean: {clean_acc*100:.2f}% | "
              f"FGSM: {fgsm_acc*100:.2f}% | PGD: {pgd_acc*100:.2f}%")

        with torch.no_grad():
            model.eval()
            dummy = torch.randn(1, 3, 32, 32).to(device)
            out = model(dummy)
            assert out.shape == (1, 10), "Output shape must be (1, 10)"

        export_name = f"robust_model_epoch_{ep:02d}.pt"
        torch.save(model.cpu().state_dict(), export_name)
        print(f"✔ Model saved to {export_name}")
        model.to(device)

# ---------- Entry ----------
if __name__ == "__main__":
    args = parse_args()
    train(args)
