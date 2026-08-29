"""
Barre varios umbrales de confianza SIN correr el cascade+CNN una vez por
umbral (evaluate.py si lo hace, y cada corrida tarda minutos). Corre el
cascade+CNN UNA sola vez por escena con un piso bajo de confianza, cachea
todos los candidatos con su probabilidad calibrada, y aplica NMS + umbral
varias veces sobre esos mismos candidatos.

Es equivalente al pipeline real: pipeline.detect() hoy filtra por --conf
ANTES de aplicar NMS, pero como pipeline.nms() es un NMS voraz por score
(siempre gana el candidato de mayor probabilidad en cada cluster que se
superponga), filtrar-y-despues-NMS da exactamente el mismo resultado que
NMS-sobre-todo-y-despues-filtrar -- el umbral solo decide si el candidato
que "gano" el cluster queda o no, no cambia quien gana.

Uso:
    python sweep_conf.py
    python sweep_conf.py --thresholds 0.9,0.95,0.99,0.995,0.998,0.9995,0.9998,0.9999
    python sweep_conf.py --floor 0.85   # umbral minimo cacheado (default 0.90)
"""

import argparse
import time

import cv2
import torch

from cnn_reclassifier.model import preprocess_bgr
from cnn_reclassifier.pipeline import DEVICE, WallyDetector, nms
from cnn_reclassifier.prepare_dataset import gather_wally_boxes, held_out_split
from evaluate import score

DEFAULT_THRESHOLDS = [0.90, 0.95, 0.99, 0.995, 0.998, 0.9995, 0.9998, 0.9999]


def raw_candidates(det, image_path, floor, max_side=2600, scale_factor=1.03, min_size=16, max_size=400):
    """[(x, y, w, h, prob), ...] en coords originales, con prob > floor. Una sola
    pasada de cascade + CNN (igual costo que un evaluate.py, sin importar cuantos
    umbrales se prueben despues)."""
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    img_s = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

    candidates = det.cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=0,
        minSize=(min_size, min_size), maxSize=(max_size, max_size),
    )

    kept = []
    inv = 1.0 / scale
    batch_size = 4096
    with torch.no_grad():
        batch_tensors, batch_boxes = [], []

        def flush():
            if not batch_tensors:
                return
            batch = torch.stack(batch_tensors).to(DEVICE)
            logits = det.model(batch) / det.temperature
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            for (x, y, w, h), p in zip(batch_boxes, probs.tolist()):
                if p > floor:
                    kept.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv), p))
            batch_tensors.clear()
            batch_boxes.clear()

        for (x, y, w, h) in candidates:
            crop = img_s[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            batch_tensors.append(preprocess_bgr(crop))
            batch_boxes.append((int(x), int(y), int(w), int(h)))
            if len(batch_tensors) >= batch_size:
                flush()
        flush()
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--floor", type=float, default=0.90, help="umbral minimo cacheado (no se puede barrer por debajo)")
    args = ap.parse_args()
    thresholds = sorted(float(t) for t in args.thresholds.split(","))
    if thresholds[0] < args.floor:
        raise SystemExit(f"--floor ({args.floor}) tiene que ser <= el menor umbral pedido ({thresholds[0]})")

    print("Cargando pipeline Haar+CNN...")
    det = WallyDetector()

    all_boxes = gather_wally_boxes()
    train_boxes, test_boxes = held_out_split(all_boxes)
    scenes = [(p, gt, "train") for p, gt in train_boxes.items()] + [(p, gt, "test") for p, gt in test_boxes.items()]
    print(f"{len(scenes)} escenas ({len(train_boxes)} train, {len(test_boxes)} test) -- cacheando candidatos (piso conf>{args.floor})...")

    cache = {}  # img_path -> (gt, split, [(x,y,w,h,prob), ...])
    t0 = time.time()
    for img_path, gt, split in scenes:
        cands = raw_candidates(det, img_path, args.floor)
        cache[img_path] = (gt, split, cands)
    print(f"Listo en {time.time()-t0:.0f}s. Barriendo {len(thresholds)} umbrales sobre los candidatos cacheados...\n")

    hdr = f"{'conf':>10} | {'TRAIN recall':>13} {'TRAIN fp':>9} | {'TEST recall':>12} {'TEST fp':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in thresholds:
        agg = {"train": [0, 0, 0], "test": [0, 0, 0]}  # n, hits, fp
        for img_path, (gt, split, cands) in cache.items():
            filtered = [c for c in cands if c[4] > t]
            dets = nms(filtered, iou_thresh=0.3)
            hit, _, _, fp = score(dets, gt)
            agg[split][0] += 1
            agg[split][1] += 1 if hit else 0
            agg[split][2] += fp
        tr, te = agg["train"], agg["test"]
        tr_s = f"{tr[1]}/{tr[0]}" if tr[0] else "-"
        te_s = f"{te[1]}/{te[0]}" if te[0] else "-"
        print(f"{t:>10} | {tr_s:>13} {tr[2]:>9} | {te_s:>12} {te[2]:>8}")


if __name__ == "__main__":
    main()
