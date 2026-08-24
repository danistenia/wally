"""
Pipeline completo de 2 etapas del paper "Where's Wally?".

    Etapa 1 (Haar-cascade): propone candidatos. Es bueno descartando el fondo
             (diciendo donde NO esta Wally) pero deja muchos falsos positivos.
    Etapa 2 (CNN):          re-clasifica cada candidato y solo conserva los que
             son Wally con probabilidad > 90%.

Uso programatico:
    from cnn_reclassifier.pipeline import WallyDetector
    det = WallyDetector()
    boxes = det.detect("una_escena.jpg")        # [(x, y, w, h, prob), ...]
    det.annotate("una_escena.jpg", "salida.jpg")
"""

import os

import cv2
import torch

from cnn_reclassifier.model import WallyCNN, preprocess_bgr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

DEFAULT_CASCADE = os.path.join(SRC, "data_cascade_HAAR", "cascade.xml")
DEFAULT_MODEL = os.environ.get("WALLY_MODEL", os.path.join(HERE, "wally_cnn.pt"))
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def nms(boxes, iou_thresh=0.3):
    """Non-maximum suppression. boxes = [(x, y, w, h, score), ...]."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        bx, by, bw, bh, _ = best
        remaining = []
        for b in boxes:
            x, y, w, h, _ = b
            x1, y1 = max(bx, x), max(by, y)
            x2, y2 = min(bx + bw, x + w), min(by + bh, y + h)
            iw, ih = max(0, x2 - x1), max(0, y2 - y1)
            inter = iw * ih
            union = bw * bh + w * h - inter
            iou = inter / union if union else 0
            if iou < iou_thresh:
                remaining.append(b)
        boxes = remaining
    return keep


class WallyDetector:
    def __init__(self, cascade_path=DEFAULT_CASCADE, model_path=DEFAULT_MODEL):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise FileNotFoundError(f"No pude cargar el cascade: {cascade_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No existe el modelo CNN: {model_path}. "
                "Entrena primero con: python -m cnn_reclassifier.train"
            )
        ck = torch.load(model_path, map_location="cpu")
        self.model = WallyCNN()
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.model.to(DEVICE)
        self.threshold = ck.get("threshold", 0.90)
        self.temperature = ck.get("temperature", 1.0)

    def detect(
        self,
        image_path,
        conf=None,
        max_side=2600,
        scale_factor=1.03,
        min_neighbors=0,
        min_size=16,
        max_size=400,
    ):
        """Devuelve las detecciones de Wally: [(x, y, w, h, prob), ...] en coords
        de la imagen ORIGINAL."""
        conf = self.threshold if conf is None else conf
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"No pude leer la imagen: {image_path}")
        H, W = img.shape[:2]

        # 1) escala manejable para el cascade (las escenas pueden ser enormes)
        scale = min(1.0, max_side / max(H, W))
        img_s = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img
        gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

        # 2) etapa 1: candidatos del Haar-cascade
        candidates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
            maxSize=(max_size, max_size),
        )

        # 3) etapa 2: la CNN re-clasifica los candidatos, en lotes STREAMING (no
        # se junta la lista completa de tensores primero: con cascades chicos
        # (24x24) una escena densa puede proponer cientos de miles de
        # candidatos, y precalcular todos los tensores antes de batchear se
        # queda sin memoria).
        kept = []
        inv = 1.0 / scale
        batch_size = 4096
        with torch.no_grad():
            batch_tensors, batch_boxes = [], []

            def flush():
                if not batch_tensors:
                    return
                batch = torch.stack(batch_tensors).to(DEVICE)
                logits = self.model(batch) / self.temperature
                probs = torch.softmax(logits, dim=1)[:, 1].cpu()
                for (x, y, w, h), p in zip(batch_boxes, probs.tolist()):
                    if p > conf:
                        kept.append(
                            (int(x * inv), int(y * inv), int(w * inv), int(h * inv), p)
                        )
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

        return nms(kept)

    def annotate(self, image_path, out_path, conf=None, show_rejected=False, **kw):
        """Corre la deteccion y guarda la imagen con las cajas dibujadas."""
        img = cv2.imread(image_path)
        boxes = self.detect(image_path, conf=conf, **kw)
        for (x, y, w, h, prob) in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), max(2, w // 20))
            cv2.putText(
                img,
                f"Wally {prob*100:.0f}%",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, w / 120),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(out_path, img)
        return boxes
