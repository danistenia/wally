"""
Arma el dataset de la CNN (cnn_dataset/) usando una lista EXPLICITA de
escenas de entrenamiento, en vez de "todo lo que no este en HELD_OUT_TEST"
(que es lo que hace cnn_reclassifier.prepare_dataset normalmente).

Util para reproducir el dataset exacto que uso una version historica del
modelo (ej. "las mismas 30 escenas de v1, pero sacando la 22 porque ahora
es held-out"), sin tener que mover/borrar archivos de original-images/.

Uso:
    python prepare_dataset_scenes.py --scenes 1,2,3,4,5,18,19,20
    python prepare_dataset_scenes.py --scenes-file escenas_v1.txt
"""

import argparse
import glob
import os

from cnn_reclassifier.prepare_dataset import (
    NEG_OUT_DIR,
    OUT_DIR,
    POS_DIR,
    extract_positives,
    extract_random_negatives,
    gather_wally_boxes,
    mine_cascade,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", help="lista de stems separados por coma, ej. 1,2,3,18,19")
    ap.add_argument("--scenes-file", help="archivo con un stem por linea (alternativa a --scenes)")
    ap.add_argument("--no-hard", action="store_true", help="omitir hard negative mining")
    args = ap.parse_args()

    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    elif args.scenes_file:
        wanted = {l.strip() for l in open(args.scenes_file) if l.strip()}
    else:
        raise SystemExit("Falta --scenes o --scenes-file")

    all_boxes = gather_wally_boxes()
    train_boxes = {
        p: b for p, b in all_boxes.items()
        if os.path.splitext(os.path.basename(p))[0] in wanted
    }
    found = {os.path.splitext(os.path.basename(p))[0] for p in train_boxes}
    missing = wanted - found
    if missing:
        print(f"AVISO: no se encontraron estas escenas (sin imagen o sin etiqueta): {sorted(missing)}")

    print(f"Usando {len(train_boxes)} escenas de entrenamiento: {sorted(found, key=lambda x: (len(x), x))}")

    for d in (POS_DIR, NEG_OUT_DIR):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.png")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    print("Extrayendo positivos (con augmentation)...")
    n_pos = extract_positives(train_boxes)
    print(f"  {n_pos} positivos")

    print("Extrayendo negativos aleatorios...")
    n_neg = extract_random_negatives(train_boxes)
    print(f"  {n_neg} negativos aleatorios")

    if not args.no_hard:
        print("Hard negative mining con el Haar-cascade...")
        n_pos_c, n_hard = mine_cascade(train_boxes)
        print(f"  {n_pos_c} positivos extra, {n_hard} hard negatives")

    total_pos = len(glob.glob(os.path.join(POS_DIR, "*.png")))
    total_neg = len(glob.glob(os.path.join(NEG_OUT_DIR, "*.png")))
    print(f"\nDataset listo en {OUT_DIR}: {total_pos} positivos, {total_neg} negativos")


if __name__ == "__main__":
    main()
