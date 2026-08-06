"""
Entrena la CNN reclasificadora (etapa 2 del paper).

- Lee positivos/negativos de cnn_dataset/ (ver prepare_dataset.py).
- Normaliza pixeles a [0, 1] y redimensiona a 44x44 (paper 4.3).
- Split 80/20 (paper).
- Adam + cross-entropy (= sparse categorical cross-entropy del paper).
- Maneja el desbalance de clases con pesos inversos a la frecuencia.
- Reporta accuracy / precision / recall / F1 sobre validacion, con el umbral
  del paper: solo cuenta como Wally si prob(wally) > 0.90.
- Guarda el mejor modelo (por F1) en wally_cnn.pt.

Uso:
    python -m cnn_reclassifier.train                 # 30 epocas por defecto
    python -m cnn_reclassifier.train --epochs 12     # como el paper
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cnn_reclassifier.model import INPUT_SIZE, WallyCNN

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

DATA_DIR = os.environ.get("WALLY_DATA_DIR", os.path.join(SRC, "cnn_dataset"))
MODEL_PATH = os.environ.get("WALLY_MODEL", os.path.join(HERE, "wally_cnn.pt"))
WALLY_THRESHOLD = 0.90  # paper: "Only Wally objects with a probability over 90%"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


from cnn_reclassifier.model import np_to_tensor, preprocess_bgr  # noqa: E402


def load_tensor(path):
    """Lee imagen de disco -> tensor float32 [3,44,44] normalizado."""
    img = cv2.imread(path)
    return preprocess_bgr(img)


class WallyDataset(Dataset):
    """Precarga todo en memoria (dataset chico) para no leer disco cada epoca."""

    def __init__(self, samples):
        self.tensors = [load_tensor(p) for p, _ in samples]
        self.labels = [lab for _, lab in samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tensors[i], self.labels[i]


def build_splits():
    pos = [(p, 1) for p in glob.glob(os.path.join(DATA_DIR, "positives", "*.png"))]
    neg = [(p, 0) for p in glob.glob(os.path.join(DATA_DIR, "negatives", "*.png"))]
    if not pos or not neg:
        raise SystemExit(
            f"No hay dataset en {DATA_DIR}. Corre antes: "
            "python -m cnn_reclassifier.prepare_dataset"
        )

    # split estratificado 80/20
    def split(items):
        random.shuffle(items)
        k = int(0.8 * len(items))
        return items[:k], items[k:]

    ptr, pte = split(pos)
    ntr, nte = split(neg)
    train = ptr + ntr
    test = pte + nte
    random.shuffle(train)
    random.shuffle(test)
    return train, test, len(pos), len(neg)


@torch.no_grad()
def evaluate(model, loader, device):
    """TP/TN/FP/FN con el umbral de 90% + metricas del paper."""
    model.eval()
    tp = tn = fp = fn = 0
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1].cpu()
        pred = (probs > WALLY_THRESHOLD).long()
        y = y.long()
        tp += int(((pred == 1) & (y == 1)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test, n_pos, n_neg = build_splits()
    print(f"Dataset: {n_pos} positivos, {n_neg} negativos")
    print(f"Train: {len(train)}  Test: {len(test)}  (device={device})")

    train_loader = DataLoader(WallyDataset(train), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(WallyDataset(test), batch_size=args.batch)

    model = WallyCNN().to(device)

    # peso inverso a la frecuencia para compensar el desbalance
    w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float32)
    w = (w / w.sum() * 2).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        m = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:2d} | loss {running/len(train):.4f} | "
            f"acc {m['acc']:.3f} prec {m['prec']:.3f} rec {m['rec']:.3f} f1 {m['f1']:.3f}"
        )
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": INPUT_SIZE,
                    "threshold": WALLY_THRESHOLD,
                    "metrics": m,
                },
                MODEL_PATH,
            )

    print(f"\nMejor F1: {best_f1:.3f}")
    print(f"Modelo guardado en {MODEL_PATH}")
    # tabla final estilo paper (Table 1)
    ck = torch.load(MODEL_PATH, map_location="cpu")
    m = ck["metrics"]
    print("\n--- Resultados en validacion (umbral 90%) ---")
    print(f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(
        f"Accuracy={m['acc']*100:.2f}%  Precision={m['prec']*100:.2f}%  "
        f"Recall={m['rec']*100:.2f}%  F1={m['f1']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
