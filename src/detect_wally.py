"""
Encuentra a Wally en una imagen usando el framework de 2 etapas del paper
(Haar-cascade + CNN reclasificadora).

Ejemplos:
    python detect_wally.py --image original-images/1.jpg
    python detect_wally.py --image /ruta/a/escena.jpg --out resultado.jpg --conf 0.9

Si no encuentra a Wally lo dice explicitamente (recuerda: la etapa Haar es
buena diciendo donde NO esta; la CNN confirma donde SI).
"""

import argparse
import os

from cnn_reclassifier.pipeline import WallyDetector


def main():
    ap = argparse.ArgumentParser(description="Encuentra a Wally en una imagen")
    ap.add_argument("--image", required=True, help="ruta de la escena a analizar")
    ap.add_argument("--out", default=None, help="ruta de la imagen anotada de salida")
    ap.add_argument(
        "--conf",
        type=float,
        default=None,
        help="umbral de confianza calibrada de la CNN (por defecto 0.9998, ver PAPER_REPRODUCTION.md)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"No existe la imagen: {args.image}")

    out = args.out
    if out is None:
        base, ext = os.path.splitext(os.path.basename(args.image))
        out = f"{base}_wally{ext or '.jpg'}"

    det = WallyDetector()
    boxes = det.annotate(args.image, out, conf=args.conf)

    if boxes:
        print(f"Wally encontrado ({len(boxes)} deteccion/es):")
        for (x, y, w, h, prob) in boxes:
            print(f"  - en (x={x}, y={y}, w={w}, h={h}) con {prob*100:.1f}% de confianza")
    else:
        print("No se encontro a Wally en esta imagen (o por debajo del umbral).")
    print(f"Imagen anotada guardada en: {out}")


if __name__ == "__main__":
    main()
