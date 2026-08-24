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

## Etapa 1: el Haar-cascade (reentrenado)

La primera versión de este repo reutilizaba un cascade ya entrenado (64×64,
4-5 stages) sin retocarlo, porque OpenCV 4.x no trae `opencv_traincascade` en
el paquete pip. Esa versión funcionaba razonablemente en las escenas
originales (1-19) pero **fallaba por completo** en escenas agregadas después
(20-32): Wally mide ahí 14-32px reales (vs. 64px fijos en las originales), y
una ventana de cascade de 64px no puede representar un objeto más chico que
sí misma con suficiente detalle.

Diagnóstico (ver historial de la sesión que hizo este cambio): sobre las 12
escenas densas, el cascade viejo proponía candidatos con ~0 IoU contra el
Wally real en 11 de 12 — no era falta de datos genérica, era un desajuste de
escala entre el cascade y el tamaño real del objeto.

Solución: se compiló `opencv_traincascade`/`opencv_createsamples` desde el
código fuente de OpenCV 3.4 (siguen en el repo de OpenCV, solo que ya no se
distribuyen en el paquete pip) usando un contenedor Docker, y se reentrenó el
cascade con:

- **ventana 24×24** (antes 64×64) — para poder representar a Wally en el
  régimen de escala donde el cascade viejo era ciego.
- **1317 positivos** generados con jitter (padding/offset/flip/brillo
  aleatorio) a partir de las ~30 instancias reales de Wally, más **3510
  negativos** de `negatives/` y de zonas de las escenas sin Wally.
- **13 stages** (BOOST/GAB, `featureType=HAAR`, `mode=ALL`), entrenamiento
  detenido por falta de positivos "limpios" para la stage 14 (esperable: cada
  stage descarta los positivos que ya rechaza, y con pocos datos reales el
  pool se agota) — 13 stages ya alcanzan el objetivo.

El cascade nuevo vive en `src/data_cascade_HAAR/cascade.xml` (el viejo queda
como referencia en `cascade_old_64x64.xml`). El dataset y los scripts usados
para reentrenarlo están en `src/prepare_cascade_dataset.py` y
`src/cascade_dataset/` (regenerable, no está pensado para vivir en git).

Para reproducir el reentrenamiento del cascade (tarda: el training completo
tomó varias horas en esta sesión):

```bash
# 1) construir la imagen con opencv_traincascade/opencv_createsamples
#    (no vienen en el paquete pip; hay que compilarlos de OpenCV 3.4 fuente)
docker build -t wally-cascade-tools docker/opencv-traincascade

# 2) armar el dataset (positivos jitterizados + negativos)
cd src && python prepare_cascade_dataset.py

# 3) empaquetar el .vec y entrenar (desde la raiz del repo)
docker run --rm -v "$(pwd)/src/cascade_dataset:/wally/cascade_dataset" \
  wally-cascade-tools bash -c "
    cd /wally/cascade_dataset &&
    /build/opencv/build/bin/opencv_createsamples -info info.dat -vec positives.vec -num <N_POSITIVOS> -w 24 -h 24 &&
    /build/opencv/build/bin/opencv_traincascade -data classifier -vec positives.vec -bg bg.txt \
      -numPos <NUMPOS> -numNeg <NUMNEG> -numStages 15 -w 24 -h 24 \
      -featureType HAAR -mode ALL -minHitRate 0.995 -maxFalseAlarm 0.5
  "

# 4) usar el resultado
cp src/cascade_dataset/classifier/cascade.xml src/data_cascade_HAAR/cascade.xml
```

Si `traincascade` se corta antes de llegar a `numStages` (por ejemplo, se
queda sin positivos "limpios" en una escena muy chica como esta), se puede
volver a invocar el mismo comando con `-numStages <n_stages_ya_completados>`:
detecta los `stageN.xml` existentes y solo empaqueta el `cascade.xml` final,
sin reentrenar desde cero.

- `info.dat`, labels YOLO (`original-images_yolo_labels*/`), carpeta
  `negatives/`, el labeler y `checker.py` siguen siendo la fuente de verdad
  de las anotaciones.

## Qué se agregó para completar el paper (etapa 2)

```
src/cnn_reclassifier/
  model.py            # arquitectura CNN (paper §4.3) + preprocesamiento
  prepare_dataset.py  # arma el dataset: positivos, negativos y hard negatives
  train.py            # entrena la CNN y reporta accuracy/precision/recall/F1
  mine_round2.py      # hard negative mining iterativo
  pipeline.py         # framework completo de 2 etapas (cascade -> CNN -> NMS)
  wally_cnn.pt        # modelo entrenado (listo para usar)
src/detect_wally.py           # CLI: encuentra a Wally en una imagen cualquiera
src/evaluate.py                # evalua el pipeline sobre las escenas etiquetadas
src/prepare_cascade_dataset.py # arma el dataset para reentrenar el cascade (etapa 1)
src/cascade_dataset/           # dataset + cascade.xml generados (regenerable)
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
  ("that the same false positives were not detected"). Se corrieron 3 rondas
  hasta ahora: la 3ra (`--round 3`, contra el cascade+CNN calibrados de esta
  ronda) agregó 3029 hard negatives nuevos (10419 negativos totales) y bajó
  los falsos positivos in-sample de 417 a 49 sin perder recall (ver
  "Resultados"). Cada ronda nueva vale la pena repetirla tras cualquier
  cambio al modelo (recalibrar, reentrenar), porque los falsos positivos que
  produce el pipeline cambian con el modelo.

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

# 4) (opcional) recalibrar la confianza tras reentrenar (ver seccion de
#    calibracion mas abajo) — todo reentrenamiento resetea la temperatura a 1.0
python -m cnn_reclassifier.calibrate

# 5) buscar a Wally en una imagen
python detect_wally.py --image original-images/9.jpg --out resultado.jpg --conf 0.9998
```

El umbral `--conf` controla el compromiso recall/falsos-positivos. El paper usa
**0.90**, pero con el cascade nuevo (ventana 24×24) eso deja demasiado ruido:
al proponer candidatos en un rango de escalas mucho más amplio, la cantidad de
candidatos por escena es mucho mayor que con el cascade de 64×64, así que un
0.11% de falsos positivos a nivel de crop (F1 99.7% en validación) se traduce
en cientos de falsos positivos por escena. El default actual del modelo
guardado es **0.9998**, ya sobre probabilidades **calibradas** y con una 3ra
ronda de hard-negative mining (ver abajo).

### Calibración de la confianza (temperature scaling)

Motivación (ver TODO más abajo): con el softmax sin calibrar casi todo
candidato salía ~100% de confianza — el modelo estaba "seguro" tanto en los
aciertos como en la mayoría de los falsos positivos, así que el número solo
no alcanzaba para discriminarlos y el umbral tenía que elegirse a los
tumbos entre 0.90 (5012 FP) y 0.9999 (438 FP), sin puntos intermedios útiles.

Se implementó **temperature scaling** (Guo et al., 2017): un único escalar
`T` que divide los logits antes del softmax, ajustado por descenso de
gradiente (LBFGS, minimizando NLL) sobre el 20% de validación que separa
`train.py`. No cambia el ranking de las predicciones — no es una técnica de
recall/precision — sólo "achata" el softmax para que el número se lea como
una probabilidad real y el umbral tenga más granularidad para elegir.

```bash
python -m cnn_reclassifier.calibrate   # ajusta T y lo guarda en wally_cnn.pt
```

Resultado: `T = 1.499`. Sobre el set de validación (crops, mismo tipo de
dato con el que se entrena) el ECE ya era bajísimo antes de calibrar
(0.13%) — ahí el modelo no está mal calibrado, el problema es de dominio:
los candidatos reales del cascade sobre una escena completa (bordes,
personajes parecidos a Wally) son más difíciles que el set de validación e
igual salían saturados. Con `T=1.499` medido sobre candidatos reales de una
escena (`17.jpg`), el número de candidatos que superan 0.9999 baja de 290 a
94 — la calibración sí discrimina en el régimen que importa (inferencia real),
aunque no lo mida el ECE del set de validación.

Efecto práctico: el barrido de umbrales que antes saltaba de 0.90 (5012 FP)
a 0.9999 (438 FP) sin puntos intermedios, ahora es gradual. Tabla actualizada
tras la 3ra ronda de hard-negative mining (ver sección "Dataset y hard
negative mining"), que es la que de verdad bajó el ruido — la calibración
sola (sin la ronda 3) ya daba puntos intermedios, pero con el modelo viejo
el mejor punto de recall completo seguía costando ~1000 FP:

| conf (calibrado) | recall | FP totales |
|---|---|---|
| 0.90 | 30/30 | 1454 |
| 0.95 | 30/30 | 926 |
| 0.99 | 30/30 | 331 |
| 0.995 | 30/30 | 224 |
| **0.998** | **30/30** | **136** |
| 0.9995 | 30/30 | 66 |
| **0.9998 (default)** | **30/30** | **49** |
| 0.9999 | 27/30 (falla 25.jpg, 30.jpg, 31.jpg) | 38 |

`0.9998` quedó como default: es el punto más estricto que sigue dando
recall completo (30/30, **incluye `25.jpg`**, el caso límite de 14px que
antes fallaba siempre). Subir a 0.9999 gana algo de precisión pero pierde
3 escenas — no vale la pena.

**Importante:** cualquier reentrenamiento de la CNN (`train.py`) resetea
`temperature` a `1.0` en el checkpoint — hay que correr `calibrate.py` de
nuevo después, y probablemente re-elegir el umbral por defecto con el mismo
barrido (`evaluate.py` + filtrar detecciones ya calculadas a distintos
`conf`, como se hizo acá) porque la escala de confianza cambia con `T`.

## Resultados

Sobre las 30 escenas etiquetadas del repo (in-sample; ver más abajo por qué
no es aún un test set real):

| | Cascade 64×64 + CNN original | Cascade 24×24 + CNN reentrenada (ronda 3 + calibrada) |
|---|---|---|
| Recall | 16/30 (ceiling documentado: 14/17 en las escenas originales) | **30/30** a `conf=0.9998` (default) — incluye `25.jpg` |
| Escenas densas (20-32) | 0/12 — el cascade no proponía nada cerca de Wally | **12/12** |
| Falsos positivos totales | no medido sistemáticamente | **49** (`conf=0.9998`, default) vs. 1454 (`conf=0.90`) |

(Confianza calibrada con temperature scaling + 3ra ronda de hard-negative
mining, ver secciones arriba — por eso estos números difieren de versiones
previas de este documento, que llegaban como mucho a 29/30 con 417-438 FP.)

`25.jpg` (el Wally de 14px, antes el único caso que fallaba siempre) ahora
se detecta correctamente — no por un cambio dirigido a ese caso puntual,
sino como efecto colateral de la ronda 3: al entrenar con más hard
negatives el modelo separa mejor Wally de sus falsos positivos en general,
lo que también subió la probabilidad del candidato correcto en esa escena.

Métricas de la CNN sobre validación (crops, umbral 90%): accuracy 99.73%,
precisión 99.78%, recall 99.82%, F1 99.80%, sobre un dataset de 22370
positivos / 10419 negativos (incluye hard negative mining con
`mine_cascade` + 3 rondas de `mine_round2.py`). El paper reporta, sobre 12
escenas de test: recall 84.61% y F1 78.54% con su modelo custom (detectó a
Wally en 10/12).

**Importante:** estos números son **in-sample** — las 30 escenas usadas para
medir son las mismas que alimentan el entrenamiento de la CNN (el cascade no
usa etiquetas para entrenar per se, pero sus positivos jitterizados sí salen
de estas escenas). Hay un mecanismo de test set held-out ya armado
(`HELD_OUT_TEST` en `cnn_reclassifier/prepare_dataset.py`, hoy `{"17", "26"}`)
que excluye esas escenas del dataset de entrenamiento, pero **`17.jpg` y
`26.jpg` todavía no están etiquetadas** — son las dos escenas nuevas que
motivaron esta ronda de mejoras. Etiquetarlas (con el labeler o labelme +
`labelme_to_yolo.py`) haría que `evaluate.py` reporte una sección TEST real.

### TODO / próximos pasos

Recall ya está resuelto (30/30 in-sample, ver arriba). Lo que sigue es
bajar los falsos positivos aún más, en orden de ROI esperado:

- [ ] **Top-1 por confianza en `detect()`** (o top-N configurable) en vez de
      devolver todo lo que supera el umbral. Es el cambio de mayor ROI
      pendiente: "¿Dónde ESTÁ Wally?" es singular por escena, así que quedarse
      con la detección más confiada elimina la mayoría de los falsos
      positivos sin reentrenar nada. Motivado por las pruebas manuales sobre
      `17.jpg`, `26.jpg` y `chicken_love_you.jpeg` (8-27 detecciones por
      imagen, la mayoría ruido sobre personajes a rayas rojo/blanco).
- [x] **Calibrar la confianza** — temperature scaling (`cnn_reclassifier/calibrate.py`,
      `T=1.499`), ver sección "Calibración de la confianza" arriba.
- [x] **Otra ronda de hard-negative mining** (`mine_round2.py --round 3`)
      contra el cascade+CNN calibrados de esta ronda: 3029 hard negatives
      nuevos, reentrenar + recalibrar. Resultado: recall **30/30** (recupera
      `25.jpg`) con **49 FP** totales a `conf=0.9998` (antes 417 FP a 29/30).
      Fue el cambio de mayor impacto de esta tanda — más que la calibración
      sola.
- [ ] **Top-1 por confianza en `detect()`** (o top-N configurable) en vez de
      devolver todo lo que supera el umbral. Con 49 FP totales en 30 escenas
      (~1.6 por escena) el ROI de esto bajó bastante tras la ronda 3, pero
      sigue siendo relevante para imágenes fuera de dominio (ver
      `chicken_love_you.jpeg` abajo).
- [ ] **Etiquetar `17.jpg` y `26.jpg`** para tener un test set held-out real
      (`HELD_OUT_TEST` en `prepare_dataset.py` ya las excluye del
      entrenamiento, falta anotarlas con el labeler o labelme).

#### Sanity check informal: `chicken_love_you.jpeg`

No es un control negativo limpio: es un póster "encuentra 10 personajes"
(Dwight Schrute, **Wally**, Titanic, The Dude, Pickle Rick, ...) que sí tiene
un Wally-pollo real escondido. Sigue sin anotarse cuál caja es la correcta,
así que este chequeo es solo informal:

- Antes de la ronda 3 (2026-08-23): 8 detecciones a `conf=0.998`.
- Después de la ronda 3 (2026-08-24), con el default nuevo `conf=0.9998`:
  **0 detecciones** — el mining hizo que todo lo que antes pasaba el umbral
  en esta imagen (personajes a rayas, ninguno confirmado como el
  Wally-pollo real) ahora quede por debajo. A `conf=0.9` reaparecen 9
  candidatos, uno de ellos en la misma zona que antes
  (~x=784,y=612 vs. ~x=773,y=614), así que probablemente sea supresión por
  umbral y no una regresión real — pero sigue sin poder confirmarse sin
  anotar la caja correcta.
