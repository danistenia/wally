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
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from cnn_reclassifier.pipeline import DEFAULT_CASCADE, DEFAULT_MODEL, WallyDetector
from cnn_reclassifier.prepare_dataset import gather_wally_boxes, held_out_split

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SRC_DIR, "eval_runs.jsonl")

DEFAULT_CONF = 0.9998  # ver PAPER_REPRODUCTION.md: con el cascade 24x24 el
# umbral del paper (0.90) deja demasiado ruido; el modelo ya esta calibrado
# con temperature scaling (cnn_reclassifier/calibrate.py) y entrenado con una
# 3ra ronda de hard-negative mining, asi que este valor da 30/30 de recall
# in-sample con muy pocos falsos positivos (ver tabla en el doc)


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


def summarize(rows):
    """rows (de run()) -> resumen agregado para loguear/comparar entre corridas."""
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    return dict(
        n=n,
        hits=hits,
        recall=(hits / n) if n else None,
        fp_total=sum(r["fp"] for r in rows),
    )


def git_info():
    """(commit corto, dirty) del repo, o (None, None) si no se puede leer."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SRC_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=SRC_DIR, capture_output=True, text=True, check=True,
        ).stdout
        return commit, bool(status.strip())
    except Exception:
        return None, None


def model_fingerprint(cascade_path, model_path):
    """Hash corto del cascade + modelo CNN actuales: identifica el estado real
    del pipeline (mining/reentrenar) aunque no se haya hecho commit todavia."""
    h = hashlib.sha256()
    for p in (cascade_path, model_path):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            return None
    return h.hexdigest()[:12]


def log_run(conf, cascade_path, model_path, train_rows, test_rows):
    commit, dirty = git_info()
    record = dict(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        model_fingerprint=model_fingerprint(cascade_path, model_path),
        conf=conf,
        train=summarize(train_rows),
        test=summarize(test_rows),
        scenes=(
            [dict(split="train", **r) for r in train_rows]
            + [dict(split="test", **r) for r in test_rows]
        ),
    )
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n(corrida guardada en {os.path.relpath(LOG_PATH, SRC_DIR)} — ver compare_runs.py)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--cascade", default=None, help="ruta a un cascade.xml alternativo")
    ap.add_argument(
        "--sanity-image",
        default=None,
        help="imagen fuera de dominio (sin Wally) para chequear falsos positivos, ej. chicken_love_you.jpeg",
    )
    ap.add_argument(
        "--no-log", action="store_true",
        help="no guardar esta corrida en eval_runs.jsonl (por defecto se guarda)",
    )
    args = ap.parse_args()

    print("Cargando pipeline Haar+CNN...")
    det = WallyDetector(cascade_path=args.cascade) if args.cascade else WallyDetector()

    all_boxes = gather_wally_boxes()
    train_boxes, test_boxes = held_out_split(all_boxes)
    print(f"{len(all_boxes)} escenas con ground truth "
          f"({len(train_boxes)} train, {len(test_boxes)} held-out test)")

    train_rows = run(det, train_boxes, args.conf)
    test_rows = run(det, test_boxes, args.conf)
    print_table("TRAIN (in-sample)", train_rows)
    print_table("TEST (held-out, nunca visto por la CNN)", test_rows)

    if args.sanity_image:
        print(f"\n--- Sanity check ({args.sanity_image}, no deberia tener Wally) ---")
        dets = det.detect(args.sanity_image, conf=args.conf)
        print(f"{len(dets)} deteccion(es) -> {dets}")

    if not args.no_log:
        cascade_path = args.cascade or DEFAULT_CASCADE
        log_run(args.conf, cascade_path, DEFAULT_MODEL, train_rows, test_rows)


if __name__ == "__main__":
    main()
