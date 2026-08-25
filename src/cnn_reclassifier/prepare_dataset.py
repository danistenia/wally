"""
Construye el dataset de la CNN reclasificadora.

Genera tres tipos de muestras (todas se guardan en cnn_dataset/):
  positivos  -> recortes de Wally desde info.dat (imgs 1-19, coords en pixeles)
                y desde los labels YOLO (imgs 20-32, coords normalizadas).
  negativos  -> recortes aleatorios de los screenshots en negatives/ y de zonas
                de las escenas que NO contienen a Wally.
  hard neg.  -> falsos positivos del Haar-cascade (etapa 1): se corre el cascade
                sobre las escenas etiquetadas y toda deteccion que NO cae sobre
                Wally se guarda como negativo. Esto es el "hard negative mining"
                del paper: forzamos a la CNN a aprender los Wally-like que
                enganan al cascade.

Uso:
    python -m cnn_reclassifier.prepare_dataset          # todo
    python -m cnn_reclassifier.prepare_dataset --no-hard # sin hard negatives
"""

import argparse
import os
import glob
import random

import cv2
import numpy as np

# ---- rutas (relativas a src/) --------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

INFO_DAT = os.path.join(SRC, "info.dat")
ORIG_DIR = os.path.join(SRC, "original-images")
YOLO_DIR = os.path.join(SRC, "original-images_yolo_labels")
YOLO_NOHD_DIR = os.path.join(SRC, "original-images_yolo_labels_no_hd")
NEG_DIR = os.path.join(SRC, "negatives")
CASCADE_PATH = os.path.join(SRC, "data_cascade_HAAR", "cascade.xml")

# Escenas etiquetadas pero reservadas como test set: NUNCA deben aportar
# positivos/negativos/hard-negatives a la CNN. gather_wally_boxes() las
# incluye igual (para que el script de evaluacion las pueda puntuar); quien
# construye el dataset de entrenamiento debe filtrarlas con held_out_split().
HELD_OUT_TEST = {"17", "26", "33", "34"}


def held_out_split(wally_boxes):
    """Separa wally_boxes en (train, test) segun HELD_OUT_TEST (por stem de archivo)."""
    train, test = {}, {}
    for img_path, bxs in wally_boxes.items():
        stem = os.path.splitext(os.path.basename(img_path))[0]
        (test if stem in HELD_OUT_TEST else train)[img_path] = bxs
    return train, test

# El dataset generado es regenerable; por defecto vive en src/cnn_dataset.
# Se puede redirigir con WALLY_DATA_DIR (util en entornos de solo-lectura).
OUT_DIR = os.environ.get("WALLY_DATA_DIR", os.path.join(SRC, "cnn_dataset"))
POS_DIR = os.path.join(OUT_DIR, "positives")
NEG_OUT_DIR = os.path.join(OUT_DIR, "negatives")

# Parametros del cascade IDENTICOS a los del pipeline de inferencia
# (pipeline.py). Es clave que coincidan: asi los hard-negatives que ve la CNN
# en entrenamiento son exactamente los candidatos que tendra que rechazar al
# predecir. minNeighbors=0 hace que el cascade sobre-proponga (como en el paper).
DETECT = dict(
    max_side=2600,
    scale_factor=1.03,
    min_neighbors=0,
    min_size=16,
    max_size=400,
)
HARD_NEG_PER_IMAGE = 200  # tope de hard-negatives por escena (se muestrean)

random.seed(42)
np.random.seed(42)


# ---- lectura de anotaciones ----------------------------------------------
def parse_info_dat():
    """info.dat -> {ruta_imagen: [(x, y, w, h), ...]} en pixeles absolutos."""
    boxes = {}
    if not os.path.exists(INFO_DAT):
        return boxes
    with open(INFO_DAT) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            rel = parts[0]  # e.g. original-images/1.jpg
            img_path = os.path.join(SRC, rel)
            n = int(parts[1])
            vals = list(map(int, parts[2 : 2 + 4 * n]))
            bxs = [tuple(vals[i : i + 4]) for i in range(0, len(vals), 4)]
            boxes.setdefault(img_path, []).extend(bxs)
    return boxes


def parse_yolo(label_path, img_w, img_h):
    """label YOLO (clase cx cy w h normalizados) -> [(x, y, w, h)] en pixeles."""
    out = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            bw, bh = w * img_w, h * img_h
            x = cx * img_w - bw / 2
            y = cy * img_h - bh / 2
            out.append((int(round(x)), int(round(y)), int(round(bw)), int(round(bh))))
    return out


def gather_wally_boxes():
    """Junta todas las cajas de Wally de todas las fuentes -> {img_path: [boxes]}."""
    boxes = parse_info_dat()

    # Escenas ya cubiertas por info.dat (1-19, coords a mano) no se duplican.
    already = {os.path.splitext(os.path.basename(p))[0] for p in boxes}

    # El resto de las escenas en original-images/ usa labels YOLO (labelme ->
    # labelme_to_yolo.py). Se descubren automaticamente, sin rango fijo, para
    # que una escena nueva se sume sola en cuanto tenga su .txt generado.
    for img_path in sorted(glob.glob(os.path.join(ORIG_DIR, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem in already:
            continue
        lbl = os.path.join(YOLO_DIR, f"{stem}.txt")
        if not os.path.exists(lbl):
            lbl = os.path.join(YOLO_NOHD_DIR, f"{stem}.txt")
        if not os.path.exists(lbl):
            continue  # escena sin anotar todavia (ej. 26.jpg)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bxs = parse_yolo(lbl, w, h)
        if bxs:
            boxes.setdefault(img_path, []).extend(bxs)
    return boxes


# ---- utilidades -----------------------------------------------------------
def clamp_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    return x, y, w, h


def overlaps(a, b):
    """IoU de dos cajas (x, y, w, h)."""
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


def augment(crop):
    """Variantes de apariencia (flip, brillo) de un recorte ya encuadrado.
    Se usa para los positivos que YA vienen enmarcados por el cascade."""
    out = [crop, cv2.flip(crop, 1)]
    for c in list(out):
        out.append(np.clip(c.astype(np.int16) + 25, 0, 255).astype(np.uint8))
        out.append(np.clip(c.astype(np.int16) - 25, 0, 255).astype(np.uint8))
    return out


def jittered_positives(img, box, n=12):
    """Genera recortes de Wally con encuadre ALEATORIO (padding + offset).

    Esto es clave: el Haar-cascade propone a Wally con encuadres muy variables
    (a veces la cara justa, a veces con contexto). Si la CNN solo ve un encuadre
    fijo, sobreajusta y rechaza los candidatos reales. Con jitter aprende a
    reconocer a Wally sin importar como venga recortado.
    """
    x, y, w, h = box
    H, W = img.shape[:2]
    crops = []
    # variantes con jitter
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
    # variantes de apariencia sobre los recortes con jitter
    out = []
    for c in crops:
        out.append(c)
        out.append(cv2.flip(c, 1))
        if random.random() < 0.5:
            out.append(np.clip(c.astype(np.int16) + 25, 0, 255).astype(np.uint8))
    return out


# ---- extraccion -----------------------------------------------------------
def extract_positives(wally_boxes):
    os.makedirs(POS_DIR, exist_ok=True)
    idx = 0
    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        for box in bxs:
            for var in jittered_positives(img, box):
                cv2.imwrite(os.path.join(POS_DIR, f"pos_{idx:05d}.png"), var)
                idx += 1
    return idx


def extract_random_negatives(wally_boxes, per_screenshot=25, per_scene=8):
    os.makedirs(NEG_OUT_DIR, exist_ok=True)
    idx = 0

    # 1) parches de los screenshots de negatives/
    for p in glob.glob(os.path.join(NEG_DIR, "*")):
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        for _ in range(per_screenshot):
            s = random.randint(48, min(160, max(49, min(H, W))))
            if W - s <= 0 or H - s <= 0:
                continue
            x = random.randint(0, W - s)
            y = random.randint(0, H - s)
            cv2.imwrite(
                os.path.join(NEG_OUT_DIR, f"neg_rnd_{idx:05d}.png"),
                img[y : y + s, x : x + s],
            )
            idx += 1

    # 2) parches de las escenas que NO solapan a Wally
    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        got = 0
        tries = 0
        while got < per_scene and tries < per_scene * 20:
            tries += 1
            s = random.randint(48, 128)
            if W - s <= 0 or H - s <= 0:
                break
            x = random.randint(0, W - s)
            y = random.randint(0, H - s)
            cand = (x, y, s, s)
            if all(overlaps(cand, b) < 0.05 for b in bxs):
                cv2.imwrite(
                    os.path.join(NEG_OUT_DIR, f"neg_scene_{idx:05d}.png"),
                    img[y : y + s, x : x + s],
                )
                idx += 1
                got += 1
    return idx


def mine_cascade(wally_boxes):
    """Corre el Haar-cascade sobre cada escena con los MISMOS parametros del
    pipeline y reparte las detecciones:
      - IoU > 0.5 con Wally  -> positivo "enmarcado por el cascade"
      - IoU < 0.2 con Wally  -> hard negative (falso positivo del cascade)
    Devuelve (n_pos_extra, n_hard_neg).
    """
    if not os.path.exists(CASCADE_PATH):
        print("  (no hay cascade entrenado, se omite hard negative mining)")
        return 0, 0
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    os.makedirs(NEG_OUT_DIR, exist_ok=True)
    os.makedirs(POS_DIR, exist_ok=True)

    n_pos, n_neg = 0, 0
    for img_path, bxs in wally_boxes.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        scale = min(1.0, DETECT["max_side"] / max(H, W))
        img_s = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
        bxs_s = [tuple(int(round(v * scale)) for v in b) for b in bxs]
        gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        dets = cascade.detectMultiScale(
            gray,
            scaleFactor=DETECT["scale_factor"],
            minNeighbors=DETECT["min_neighbors"],
            minSize=(DETECT["min_size"], DETECT["min_size"]),
            maxSize=(DETECT["max_size"], DETECT["max_size"]),
        )

        hard = []
        for (x, y, w, h) in dets:
            cand = (int(x), int(y), int(w), int(h))
            crop = img_s[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            iou = max((overlaps(cand, b) for b in bxs_s), default=0.0)
            if iou > 0.5:  # el cascade acerto a Wally -> positivo extra
                for var in augment(crop):
                    cv2.imwrite(os.path.join(POS_DIR, f"pos_casc_{n_pos:05d}.png"), var)
                    n_pos += 1
            elif iou < 0.2:  # falso positivo -> hard negative
                hard.append(crop)

        # muestrear para no saturar con miles de parches casi iguales
        if len(hard) > HARD_NEG_PER_IMAGE:
            hard = random.sample(hard, HARD_NEG_PER_IMAGE)
        for crop in hard:
            cv2.imwrite(os.path.join(NEG_OUT_DIR, f"neg_hard_{n_neg:05d}.png"), crop)
            n_neg += 1
    return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hard", action="store_true", help="omitir hard negative mining")
    args = ap.parse_args()

    # limpiar salida previa (ignorar si el FS no permite borrar)
    for d in (POS_DIR, NEG_OUT_DIR):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.png")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    print("Juntando cajas de Wally...")
    all_boxes = gather_wally_boxes()
    wally_boxes, test_boxes = held_out_split(all_boxes)
    n_imgs = len(wally_boxes)
    n_wally = sum(len(v) for v in wally_boxes.values())
    print(f"  {n_wally} instancias de Wally en {n_imgs} escenas de entrenamiento")
    if test_boxes:
        held = ", ".join(sorted(os.path.basename(p) for p in test_boxes))
        print(f"  (excluidas del dataset, reservadas como test: {held})")

    print("Extrayendo positivos (con augmentation)...")
    n_pos = extract_positives(wally_boxes)
    print(f"  {n_pos} positivos")

    print("Extrayendo negativos aleatorios...")
    n_neg = extract_random_negatives(wally_boxes)
    print(f"  {n_neg} negativos aleatorios")

    if not args.no_hard:
        print("Hard negative mining con el Haar-cascade (params del pipeline)...")
        n_pos_c, n_hard = mine_cascade(wally_boxes)
        print(f"  {n_pos_c} positivos extra (Wally enmarcado por el cascade)")
        print(f"  {n_hard} hard negatives")

    total_pos = len(glob.glob(os.path.join(POS_DIR, "*.png")))
    total_neg = len(glob.glob(os.path.join(NEG_OUT_DIR, "*.png")))
    print(f"\nDataset listo en {OUT_DIR}")
    print(f"  positivos: {total_pos}")
    print(f"  negativos: {total_neg}")


if __name__ == "__main__":
    main()
