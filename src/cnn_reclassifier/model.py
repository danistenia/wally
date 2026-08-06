"""
CNN reclasificadora - arquitectura del paper (seccion 4.3).

    "The CNN [...] is made up of three convolutional and pooling layers with
     ReLU activation followed by a dropout, flatten, and fully connected layer.
     All convolutional layers have a kernel of size 3x3. The first convolutional
     layer extracts 32 feature maps from an image of size 44x44. The other two
     convolutional layers extract 64 feature maps. All three pooling layers
     calculate the maximum value of every 2x2 window [...]. The dropout layer
     sets input units to zero randomly at a rate of 0.2 [...]. The fully
     connected layer contains 64 units which all use ReLU activation."

Compilado con Adam y sparse categorical cross-entropy (equivalente en PyTorch:
nn.CrossEntropyLoss, que ya incluye el softmax).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

# Tamano de entrada de la CNN (paper: 44x44).
INPUT_SIZE = 44

# Nombres de clase. 0 = no wally, 1 = wally.
CLASSES = ["not_wally", "wally"]


def np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convierte un ndarray a tensor SIN usar el puente torch<->numpy.

    torch (build de PyPI) puede estar compilado contra una version de numpy
    distinta a la instalada, y torch.from_numpy falla. Copiamos los bytes crudos
    con torch.frombuffer, que no depende de ese puente.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
    return t.reshape(arr.shape)


def preprocess_bgr(bgr: np.ndarray) -> torch.Tensor:
    """Recorte BGR (como lo entrega cv2) -> tensor [3,44,44] RGB en [0,1]."""
    img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np_to_tensor(img)


class WallyCNN(nn.Module):
    """CNN ligera para reclasificar candidatos del Haar-cascade."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: Conv 3x3 -> 32 mapas -> ReLU -> MaxPool 2x2
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 2: Conv 3x3 -> 64 mapas -> ReLU -> MaxPool 2x2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bloque 3: Conv 3x3 -> 64 mapas -> ReLU -> MaxPool 2x2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 44 -> 22 -> 11 -> 5  (cada MaxPool 2x2 divide entre 2, redondeando abajo)
        feat_side = INPUT_SIZE
        for _ in range(3):
            feat_side = feat_side // 2
        flat = 64 * feat_side * feat_side

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits (el softmax lo aplica CrossEntropyLoss / softmax en inferencia)
