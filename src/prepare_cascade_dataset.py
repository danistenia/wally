"""
Construye el dataset para REENTRENAR el Haar-cascade (etapa 1) con
opencv_createsamples + opencv_traincascade.

El cascade original (data_cascade_HAAR/) se entreno hace tiempo con ~18
positivos y ventana 64x64. El diagnostico de esta sesion mostro que a esa
escala el cascade no logra proponer a Wally cuando mide <30px reales en
escenas densas (20-32): el candidato mas cercano tiene ~0 IoU con el Wally
real en 11 de 12 escenas nuevas.

Este dataset:
  - usa TODAS las escenas etiquetadas actuales, menos las held-out de test
    (ver HELD_OUT_TEST en cnn_reclassifier/prepare_dataset.py).
  - multiplica los positivos via jitter (mismo mecanismo que ya usa
    cnn_reclassifier/prepare_dataset.py para la CNN): sin esto solo hay
    ~30 instancias reales de Wally, insuficiente para boosting.
  - usa ventana CASCADE_WIN=24 (no 64) para poder representar a Wally en el
    regimen de escala donde hoy falla, y porque una ventana chica necesita
    menos datos para converger.
  - los negativos salen de negatives/ (screenshots) y de zonas de las
    escenas que NO solapan a Wally (mismo criterio que ya usa
    extract_random_negatives en cnn_reclassifier/prepare_dataset.py).

Uso:
    cd src
    python prepare_cascade_dataset.py

Genera cascade_dataset/positives/*.png, cascade_dataset/negatives/*.png,
cascade_dataset/info.dat (formato -info de opencv_createsamples) y
cascade_dataset/bg.txt (formato -bg de opencv_traincascade).
"""

import glob
import os
import random

import cv2
import numpy as np

from cnn_reclassifier.prepare_dataset import (
    gather_wally_boxes,
    held_out_split,
    overlaps,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NEG_DIR = os.path.join(HERE, "negatives")
OUT_DIR = os.path.join(HERE, "cascade_dataset")
POS_DIR = os.path.join(OUT_DIR, "positives")
NEG_OUT_DIR = os.path.join(OUT_DIR, "negatives")

CASCADE_WIN = 24  # ancho=alto de la ventana del nuevo cascade (era 64)

random.seed(42)
np.random.seed(42)


def clamp_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    return x, y, w, h


def jittered_positives(img, box, n=15):
    """Igual criterio que cnn_reclassifier.prepare_dataset.jittered_positives:
    encuadres con padding/offset aleatorio + flip + variantes de brillo, para
    que el cascade no sobreajuste a un unico recorte."""
    x, y, w, h = box
    H, W = img.shape[:2]
    crops = []
    for _ in range(n):
        pad = random.uniform(-0.05, 0.40) * max(w, h)
        dx = random.uniform(-0.15, 0.15) * w
        dy = random.uniform(-0.15, 0.15) * h
        nx, ny, nw, nh = clamp_box(
            int(x - pad + dx), int(y - pad + dy), int(w + 2 * pad), int(h + 2 * pad), W, H
        )
        if nw < 8 or nh < 8:
            continue
        crops.append(img[ny : ny + nh, nx : nx + nw])
    out = []
    for c in crops:
        out.append(c)
        out.append(cv2.flip(c, 1))
        if random.random() < 0.5:
            out.append(np.clip(c.astype(np.int16) + 25, 0, 255).astype(np.uint8))
        if random.random() < 0.5:
            out.append(np.clip(c.astype(np.int16) - 25, 0, 255).astype(np.uint8))
    return out


def extract_positives(wally_boxes):
    os.makedirs(POS_DIR, exist_ok=True)
    lines = []
    idx = 0
    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        for box in bxs:
            for crop in jittered_positives(img, box):
                h, w = crop.shape[:2]
                if w < CASCADE_WIN or h < CASCADE_WIN:
                    continue  # createsamples no puede achicar de mas
                name = f"pos_{idx:05d}.png"
                cv2.imwrite(os.path.join(POS_DIR, name), crop)
                lines.append(f"positives/{name} 1 0 0 {w} {h}")
                idx += 1
    return lines


def extract_negatives(wally_boxes, per_screenshot=60, per_scene=25):
    os.makedirs(NEG_OUT_DIR, exist_ok=True)
    paths = []
    idx = 0

    for p in glob.glob(os.path.join(NEG_DIR, "*")):
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        for _ in range(per_screenshot):
            s = random.randint(max(CASCADE_WIN, 48), min(200, max(49, min(H, W))))
            if W - s <= 0 or H - s <= 0:
                continue
            x = random.randint(0, W - s)
            y = random.randint(0, H - s)
            name = f"neg_rnd_{idx:05d}.png"
            cv2.imwrite(os.path.join(NEG_OUT_DIR, name), img[y : y + s, x : x + s])
            paths.append(f"negatives/{name}")
            idx += 1

    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        got, tries = 0, 0
        while got < per_scene and tries < per_scene * 20:
            tries += 1
            s = random.randint(max(CASCADE_WIN, 48), 160)
            if W - s <= 0 or H - s <= 0:
                break
            x = random.randint(0, W - s)
            y = random.randint(0, H - s)
            cand = (x, y, s, s)
            if all(overlaps(cand, b) < 0.05 for b in bxs):
                name = f"neg_scene_{idx:05d}.png"
                cv2.imwrite(os.path.join(NEG_OUT_DIR, name), img[y : y + s, x : x + s])
                paths.append(f"negatives/{name}")
                idx += 1
                got += 1
    return paths


def main():
    for d in (POS_DIR, NEG_OUT_DIR):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.png")):
                os.remove(f)

    print("Juntando cajas de Wally...")
    all_boxes = gather_wally_boxes()
    train_boxes, test_boxes = held_out_split(all_boxes)
    print(f"  {len(train_boxes)} escenas de entrenamiento, {len(test_boxes)} held-out test")

    print(f"Generando positivos con jitter (ventana >= {CASCADE_WIN}px)...")
    pos_lines = extract_positives(train_boxes)
    print(f"  {len(pos_lines)} positivos")

    print("Generando negativos...")
    neg_paths = extract_negatives(train_boxes)
    print(f"  {len(neg_paths)} negativos")

    with open(os.path.join(OUT_DIR, "info.dat"), "w") as f:
        f.write("\n".join(pos_lines) + "\n")
    with open(os.path.join(OUT_DIR, "bg.txt"), "w") as f:
        f.write("\n".join(neg_paths) + "\n")

    print(f"\nDataset listo en {OUT_DIR}")
    print(f"  {os.path.join(OUT_DIR, 'info.dat')}  (para -info de opencv_createsamples)")
    print(f"  {os.path.join(OUT_DIR, 'bg.txt')}     (para -bg de opencv_traincascade)")


if __name__ == "__main__":
    main()
