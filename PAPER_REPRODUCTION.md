# Reproducción del paper "Where's Wally? A machine learning approach"

Barthelmes & Vidal (2021), Bangor University. Este documento describe cómo se
recreó el framework de 2 etapas del paper con el código de este repositorio.
Todo el código nuevo vive en `src/cnn_reclassifier/` y en `src/detect_wally.py`.

## La idea del paper

El paper detecta a Wally combinando una técnica clásica con una moderna:

1. **Etapa 1 — Haar-cascade classifier (Viola-Jones).** Propone *candidatos*.
   Es bueno descartando el fondo (dice dónde **NO** está Wally) pero deja
   cientos de falsos positivos porque no distingue a Wally de personajes
   parecidos (rayas rojas/blancas).
2. **Etapa 2 — CNN ligera reclasificadora.** Toma cada candidato del cascade y
   decide *Wally / no-Wally*. Solo cuenta como Wally si la probabilidad supera
   el 90%.

La ventaja frente a una CNN end-to-end es velocidad: el cascade elimina rápido
casi todo el fondo y la CNN solo evalúa un puñado de candidatos.

## Qué ya existía en el repo (etapa 1)

- `src/data_cascade_HAAR/cascade.xml` — el Haar-cascade ya entrenado (64×64).
  Se **reutiliza tal cual**; no se reentrena (OpenCV 4.x ya no trae
  `opencv_traincascade`, y el cascade existente funciona).
- `info.dat`, labels YOLO, carpeta `negatives/`, el labeler y `checker.py`.

## Qué se agregó para completar el paper (etapa 2)

```
src/cnn_reclassifier/
  model.py            # arquitectura CNN (paper §4.3) + preprocesamiento
  prepare_dataset.py  # arma el dataset: positivos, negativos y hard negatives
  train.py            # entrena la CNN y reporta accuracy/precision/recall/F1
  mine_round2.py      # hard negative mining iterativo
  pipeline.py         # framework completo de 2 etapas (cascade -> CNN -> NMS)
  wally_cnn.pt        # modelo entrenado (listo para usar)
src/detect_wally.py   # CLI: encuentra a Wally en una imagen cualquiera
```

### Arquitectura de la CNN (fiel al paper §4.3)

Entrada 44×44 RGB, píxeles normalizados a [0,1]:

- Conv 3×3 (32 mapas) → ReLU → MaxPool 2×2
- Conv 3×3 (64 mapas) → ReLU → MaxPool 2×2
- Conv 3×3 (64 mapas) → ReLU → MaxPool 2×2
- Dropout 0.2 → Flatten → Dense 64 (ReLU) → Dense 2 (softmax)
- Optimizador Adam, pérdida cross-entropy (equivale a la "sparse categorical
  cross-entropy" del paper).

Única diferencia de implementación: el paper usa TensorFlow; aquí se usa
PyTorch (misma arquitectura).

### Dataset y hard negative mining

- **Positivos:** recortes de Wally desde `info.dat` (imágenes 1–19) y desde los
  labels YOLO (20–32). Se generan con *jitter* de encuadre (padding y offset
  aleatorios) para que la CNN reconozca a Wally sin importar cómo lo recorte el
  cascade. Esto fue clave: sin jitter, la CNN sobreajusta a un único encuadre y
  rechaza los candidatos reales.
- **Negativos:** parches de los screenshots de `negatives/`, zonas de las
  escenas sin Wally, y sobre todo **hard negatives**: se corre el cascade sobre
  las escenas y sus falsos positivos (los Wally-like que lo engañan) se agregan
  como negativos. El paper llama a esto *hard negative mining*.
- **Mining iterativo (`mine_round2.py`):** tras entrenar, se corre el pipeline
  completo y los falsos positivos que aún produce se reinyectan como negativos.
  Reentrenar con ellos reduce los falsos positivos, como busca el paper
  ("that the same false positives were not detected").

## Cómo ejecutar

```bash
cd src

# 1) (opcional) regenerar el dataset de la CNN
python -m cnn_reclassifier.prepare_dataset

# 2) (opcional) entrenar la CNN — ya hay un modelo entrenado en wally_cnn.pt
python -m cnn_reclassifier.train --epochs 25

# 3) (opcional) mining iterativo + reentrenar
python -m cnn_reclassifier.mine_round2 --conf 0.6 --cap 120 --round 3
python -m cnn_reclassifier.train

# 4) buscar a Wally en una imagen
python detect_wally.py --image original-images/9.jpg --out resultado.jpg --conf 0.99
```

El umbral `--conf` controla el compromiso recall/falsos-positivos: el paper usa
**0.90**; subirlo a **0.99** reduce mucho los falsos positivos manteniendo casi
todo el recall (recomendado para inspección visual).

## Resultados

Sobre las escenas etiquetadas del repo, el framework encuentra a Wally en
**~10–11 de 17** escenas. El **techo de la etapa 1** es 14/17: el Haar-cascade
simplemente no propone a Wally en las imágenes 3, 4 y 5 (limitación del cascade
entrenado; ninguna CNN puede recuperarlas). Es decir, estamos cerca del máximo
alcanzable con este cascade.

Métricas de la CNN sobre validación (crops, umbral 90%): accuracy ~98–99%,
precisión ~95%, recall ~94–96%, F1 ~95%. El paper reporta, sobre 12 escenas de
test: recall 84.61% y F1 78.54% con su modelo custom (detectó a Wally en 10/12).

### Cómo mejorar (trabajo futuro, como sugiere el paper)

- Reentrenar el **Haar-cascade** con más positivos para subir el techo (recuperar
  imágenes 3/4/5).
- Más datos de entrenamiento para la CNN (el dataset es chico, la principal
  limitación que menciona el paper).
- Más rondas de hard negative mining para bajar los falsos positivos.
