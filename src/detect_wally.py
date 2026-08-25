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
import glob
import os

from cnn_reclassifier.pipeline import WallyDetector

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.join(SRC_DIR, "model_versions")


def resolve_version(version):
    """--version 3 o --version round3-mining -> (cascade_path, model_path) de model_versions/."""
    matches = [
        d for d in glob.glob(os.path.join(VERSIONS_DIR, "v*"))
        if os.path.basename(d) == f"v{version}" or os.path.basename(d).startswith(f"v{version}_")
        or os.path.basename(d).endswith(f"_{version}")
    ]
    if not matches:
        raise SystemExit(
            f"No encontre la version '{version}' en {VERSIONS_DIR}. "
            "Usa 'python save_version.py --list' para ver las disponibles."
        )
    d = matches[0]
    return os.path.join(d, "cascade.xml"), os.path.join(d, "wally_cnn.pt")


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
    ap.add_argument(
        "--version",
        default=None,
        help="correr contra una version guardada (ej. '3' o 'round3-mining') en vez del modelo actual, "
        "para comparar resultados entre versiones -- ver save_version.py",
    )
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"No existe la imagen: {args.image}")

    out = args.out
    if out is None:
        base, ext = os.path.splitext(os.path.basename(args.image))
        suffix = f"_v{args.version}" if args.version else ""
        out = f"{base}_wally{suffix}{ext or '.jpg'}"

    if args.version:
        cascade_path, model_path = resolve_version(args.version)
        det = WallyDetector(cascade_path=cascade_path, model_path=model_path)
    else:
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
