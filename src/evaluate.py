"""
Evalua el pipeline Haar+CNN (cnn_reclassifier/pipeline.py) sobre las escenas
etiquetadas, separando:
  - TRAIN: escenas que alimentan el dataset de la CNN (in-sample).
  - TEST : escenas held-out (ver HELD_OUT_TEST en prepare_dataset.py), nunca
           vistas por la CNN. Es la metrica que de verdad importa.

Uso:
    cd src
    python evaluate.py
    python evaluate.py --conf 0.99
    python evaluate.py --sanity-image chicken_love_you.jpeg
"""

import argparse
import os
import time

from cnn_reclassifier.pipeline import WallyDetector
from cnn_reclassifier.prepare_dataset import gather_wally_boxes, held_out_split

DEFAULT_CONF = 0.998  # ver PAPER_REPRODUCTION.md: con el cascade 24x24 el
# umbral del paper (0.90) deja demasiado ruido; el modelo ya esta calibrado
# con temperature scaling (cnn_reclassifier/calibrate.py), asi que este valor
# se eligio sobre probabilidades calibradas, no saturadas


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def score(dets, gt_boxes, iou_thr=0.3):
    """dets = [(x,y,w,h,conf), ...] -> (hit, best_iou, best_conf, n_false_pos)."""
    best_iou, best_conf, hit = 0.0, 0.0, False
    fp = 0
    for (x, y, w, h, c) in dets:
        m = max((iou((x, y, w, h), g) for g in gt_boxes), default=0.0)
        if m > iou_thr:
            hit = True
            if m > best_iou:
                best_iou, best_conf = m, c
        else:
            fp += 1
    return hit, best_iou, best_conf, fp


def run(det, wally_boxes, conf):
    rows = []
    for img_path in sorted(wally_boxes):
        gt = wally_boxes[img_path]
        t0 = time.time()
        dets = det.detect(img_path, conf=conf)
        dt = time.time() - t0
        hit, best_iou, best_conf, fp = score(dets, gt)
        rows.append(dict(
            name=os.path.basename(img_path), n_gt=len(gt),
            hit=hit, iou=best_iou, conf=best_conf, fp=fp, time=dt,
        ))
    return rows


def print_table(title, rows):
    print(f"\n--- {title} ({len(rows)} escenas) ---")
    if not rows:
        print("  (sin escenas)")
        return
    hdr = f"{'imagen':<10} {'GT':>2} | {'hit':>4} {'iou':>5} {'conf':>5} {'fp':>3} {'seg':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['name']:<10} {r['n_gt']:>2} | "
            f"{str(r['hit']):>4} {r['iou']:.2f} {r['conf']:.2f} {r['fp']:>3} {r['time']:>5.1f}"
        )
    n = len(rows)
    found = sum(r["hit"] for r in rows)
    fp_total = sum(r["fp"] for r in rows)
    print(f"\nResumen {title}: encontrado en {found}/{n} escenas, {fp_total} falsos positivos totales")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--cascade", default=None, help="ruta a un cascade.xml alternativo")
    ap.add_argument(
        "--sanity-image",
        default=None,
        help="imagen fuera de dominio (sin Wally) para chequear falsos positivos, ej. chicken_love_you.jpeg",
    )
    args = ap.parse_args()

    print("Cargando pipeline Haar+CNN...")
    det = WallyDetector(cascade_path=args.cascade) if args.cascade else WallyDetector()

    all_boxes = gather_wally_boxes()
    train_boxes, test_boxes = held_out_split(all_boxes)
    print(f"{len(all_boxes)} escenas con ground truth "
          f"({len(train_boxes)} train, {len(test_boxes)} held-out test)")

    print_table("TRAIN (in-sample)", run(det, train_boxes, args.conf))
    print_table("TEST (held-out, nunca visto por la CNN)", run(det, test_boxes, args.conf))

    if args.sanity_image:
        print(f"\n--- Sanity check ({args.sanity_image}, no deberia tener Wally) ---")
        dets = det.detect(args.sanity_image, conf=args.conf)
        print(f"{len(dets)} deteccion(es) -> {dets}")


if __name__ == "__main__":
    main()
