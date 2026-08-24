"""
Calibra la confianza de la CNN reclasificadora (etapa 2) con temperature
scaling (Guo et al., "On Calibration of Modern Neural Networks", 2017).

El softmax de la CNN sale saturado (~100% en casi todo candidato), lo que
hace que el umbral de confianza no discrimine bien aciertos de errores por
el numero solo. Temperature scaling divide los logits por un escalar T>1
aprendido sobre un set de validacion: "aplana" el softmax sin tocar el
ranking de las predicciones (no cambia que candidato pasa un umbral dado en
terminos de orden, solo hace que el numero se lea como una probabilidad
real en vez de una senial saturada).

Uso:
    python -m cnn_reclassifier.calibrate
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn_reclassifier.model import WallyCNN
from cnn_reclassifier.train import MODEL_PATH, WallyDataset, build_splits


@torch.no_grad()
def collect_logits(model, loader, device):
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: diferencia promedio (ponderada por bin) entre confianza y accuracy."""
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.float().mean() * (bin_acc - bin_conf).abs()).item()
    return ece


def fit_temperature(logits, labels, max_iter=50, lr=0.01):
    """Ajusta un escalar T minimizando NLL de softmax(logits/T) vs labels."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bins", type=int, default=10, help="bins para el ECE")
    args = ap.parse_args()

    device = torch.device("cpu")

    # Mismo split 80/20 que train.py (misma seed): la porcion de test nunca
    # recibe gradientes, aunque train.py la usa para elegir el checkpoint de
    # mejor F1 -- es el mismo compromiso que hace el paper de calibracion
    # original al reusar el set de validacion.
    _, test, _, _ = build_splits()
    loader = DataLoader(WallyDataset(test), batch_size=64)

    ck = torch.load(MODEL_PATH, map_location="cpu")
    model = WallyCNN()
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)

    logits, labels = collect_logits(model, loader, device)

    probs_before = torch.softmax(logits, dim=1)
    ece_before = expected_calibration_error(probs_before, labels, args.bins)

    T = fit_temperature(logits, labels)

    probs_after = torch.softmax(logits / T, dim=1)
    ece_after = expected_calibration_error(probs_after, labels, args.bins)

    n_wally = int((labels == 1).sum())
    n_not = int((labels == 0).sum())
    print(f"Set de calibracion: {len(labels)} crops ({n_wally} wally, {n_not} no-wally)")
    print(f"Temperatura ajustada: T = {T:.3f}")
    print(f"ECE antes de calibrar:   {ece_before*100:.2f}%")
    print(f"ECE despues de calibrar: {ece_after*100:.2f}%")

    ck["temperature"] = T
    torch.save(ck, MODEL_PATH)
    print(f"\nGuardado T={T:.3f} en {MODEL_PATH}")
    print(
        "Nota: --conf ahora se interpreta sobre probabilidades calibradas "
        "(ya no saturan en ~100%). Puede convenir re-elegir el umbral por "
        "defecto con evaluate.py."
    )


if __name__ == "__main__":
    main()
