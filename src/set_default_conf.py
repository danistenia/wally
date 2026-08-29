"""
Actualiza el umbral de confianza por defecto que queda guardado dentro del
checkpoint de la CNN (wally_cnn.pt) -- el que usa WallyDetector.detect()
cuando se llama SIN --conf explicito (asi lo usan detect_wally.py y
cualquier otro consumidor de la API que no pase el umbral a mano).

train.py siempre resetea ese campo a 0.90 (el umbral del paper, pensado solo
para reportar metricas de validacion sobre crops) -- NO es el umbral de
despliegue elegido con evaluate.py/sweep_conf.py. Despues de elegir el
umbral real, correr esto para que el default silencioso quede sincronizado
y no vuelva a pasar que un detect_wally.py sin flags use 0.90 por accidente.

Uso:
    python set_default_conf.py --conf 0.9998
"""

import argparse

import torch

from cnn_reclassifier.pipeline import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, required=True)
    args = ap.parse_args()

    ck = torch.load(DEFAULT_MODEL, map_location="cpu")
    old = ck.get("threshold")
    ck["threshold"] = args.conf
    torch.save(ck, DEFAULT_MODEL)
    print(f"threshold del checkpoint: {old} -> {args.conf}  (guardado en {DEFAULT_MODEL})")


if __name__ == "__main__":
    main()
