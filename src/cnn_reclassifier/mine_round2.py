"""
Hard negative mining ITERATIVO (segunda ronda).

El paper busca "que los mismos falsos positivos no fueran detectados". Tras
entrenar una primera CNN, corremos el pipeline COMPLETO sobre las escenas
etiquetadas y todo lo que detecta como Wally pero NO lo es (IoU < 0.2 con el
Wally real) se agrega al dataset como hard negative. Reentrenar con estos
ejemplos elimina la mayoria de los falsos positivos.

Uso:
    python -m cnn_reclassifier.mine_round2
    python -m cnn_reclassifier.train        # reentrenar con los nuevos negativos
"""

import argparse
import glob
import os
import random

import cv2

from cnn_reclassifier.prepare_dataset import (
    NEG_OUT_DIR,
    gather_wally_boxes,
    held_out_split,
    overlaps,
)
from cnn_reclassifier.pipeline import WallyDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.6,
                    help="recolectar candidatos-Wally por encima de este umbral")
    ap.add_argument("--cap", type=int, default=120,
                    help="tope de falsos positivos por escena")
    ap.add_argument("--round", type=int, default=2, help="etiqueta de la ronda")
    args = ap.parse_args()

    os.makedirs(NEG_OUT_DIR, exist_ok=True)
    det = WallyDetector()
    wally_boxes, _ = held_out_split(gather_wally_boxes())

    start = len(glob.glob(os.path.join(NEG_OUT_DIR, f"neg_r{args.round}_*.png")))
    added = 0
    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        dets = det.detect(img_path, conf=args.conf, min_neighbors=0)
        fps = [
            (x, y, w, h)
            for (x, y, w, h, _) in dets
            if all(overlaps((x, y, w, h), b) < 0.2 for b in bxs)
        ]
        if len(fps) > args.cap:
            fps = random.sample(fps, args.cap)
        for (x, y, w, h) in fps:
            crop = img[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            cv2.imwrite(
                os.path.join(NEG_OUT_DIR, f"neg_r{args.round}_{start+added:05d}.png"),
                crop,
            )
            added += 1
    print(f"Ronda {args.round}: {added} falsos positivos agregados como hard negatives.")
    print("Ahora reentrena: python -m cnn_reclassifier.train")


if __name__ == "__main__":
    main()
