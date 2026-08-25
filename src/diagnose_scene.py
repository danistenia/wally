"""
Diagnostica una escena puntual: separa si el cuello de botella esta en la
etapa 1 (el Haar-cascade no propone NINGUN candidato cerca de Wally) o en la
etapa 2 (el cascade si propuso un candidato cerca, pero la CNN lo rechazo por
confianza baja). Sirve para decidir si hace falta reentrenar el cascade
(caro, horas) o alcanza con reentrenar la CNN (rapido, minutos).

Uso:
    python diagnose_scene.py --image original-images/17.jpg
"""

import argparse
import os

import cv2
import torch

from cnn_reclassifier.model import preprocess_bgr
from cnn_reclassifier.pipeline import DEVICE, WallyDetector
from cnn_reclassifier.prepare_dataset import gather_wally_boxes


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-side", type=int, default=2600)
    ap.add_argument("--scale-factor", type=float, default=1.03)
    ap.add_argument("--min-size", type=int, default=16)
    ap.add_argument("--max-size", type=int, default=400)
    ap.add_argument("--out", default=None, help="guardar imagen anotada con TODOS los candidatos del cascade")
    args = ap.parse_args()

    img_path = os.path.abspath(args.image)
    all_boxes = gather_wally_boxes()
    gt = None
    for p, bxs in all_boxes.items():
        if os.path.abspath(p) == img_path:
            gt = bxs
            break
    if not gt:
        raise SystemExit(f"No hay ground truth para {args.image} (etiquetala primero con labelme)")

    det = WallyDetector()
    img = cv2.imread(args.image)
    H, W = img.shape[:2]
    scale = min(1.0, args.max_side / max(H, W))
    img_s = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

    candidates = det.cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=0,
        minSize=(args.min_size, args.min_size),
        maxSize=(args.max_size, args.max_size),
    )
    gt_s = [tuple(int(round(v * scale)) for v in b) for b in gt]

    print(f"{args.image}: {len(candidates)} candidatos del cascade, {len(gt_s)} Wally(s) real(es), escala={scale:.3f}")

    for gi, g in enumerate(gt_s):
        best_iou, best_cand = 0.0, None
        for (x, y, w, h) in candidates:
            i = iou((x, y, w, h), g)
            if i > best_iou:
                best_iou, best_cand = i, (int(x), int(y), int(w), int(h))

        print(f"\nWally #{gi+1} (caja real, escalada: {g}):")
        if best_cand is None or best_iou < 0.05:
            print(f"  ETAPA 1 (cascade): NO propuso ningun candidato cerca (mejor IoU={best_iou:.2f}).")
            print("  -> el cuello de botella esta en el cascade: hace falta reentrenarlo (caro) con mas")
            print("     diversidad de pose/escala/contexto para que aprenda a proponer esta escena.")
            continue

        x, y, w, h = best_cand
        crop = img_s[y : y + h, x : x + w]
        with torch.no_grad():
            t = preprocess_bgr(crop).unsqueeze(0).to(DEVICE)
            logits = det.model(t) / det.temperature
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        print(f"  ETAPA 1 (cascade): SI propuso un candidato cerca -> IoU={best_iou:.2f}, caja={best_cand}")
        print(f"  ETAPA 2 (CNN): confianza calibrada = {prob*100:.4f}%  (umbral actual: {det.threshold*100:.2f}%)")
        if prob > det.threshold:
            print("  -> raro: con esta confianza deberia haberse detectado. Revisar NMS/umbral usado en evaluate.py.")
        else:
            print("  -> el cascade SI encontro a Wally, pero la CNN lo rechazo por confianza baja.")
            print("     el cuello de botella esta en la CNN: alcanza con sumar este tipo de ejemplo")
            print("     y reentrenar la CNN (rapido), sin tocar el cascade.")

    if args.out:
        vis = img_s.copy()
        for (x, y, w, h) in candidates:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
        for g in gt_s:
            x, y, w, h = g
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(args.out, vis)
        print(f"\nImagen anotada guardada en: {args.out} (verde=candidatos del cascade, rojo=Wally real)")


if __name__ == "__main__":
    main()
