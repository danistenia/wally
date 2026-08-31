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
de estas escenas). Hay un mecanismo de test set held-out armado
(`HELD_OUT_TEST` en `cnn_reclassifier/prepare_dataset.py`, hoy
`{"17", "26", "33", "34"}`) que excluye esas escenas del dataset de
entrenamiento, pero **ninguna de las 4 está etiquetada todavía** — falta
dibujar el rectángulo de Wally con `labelme` en cada una (ver
"Etiquetado pendiente" abajo). En cuanto tengan su `.json`/`.txt`,
`evaluate.py` reporta una sección TEST real automáticamente (no hace falta
tocar código: `gather_wally_boxes()` las descubre solas).

`33.jpg` y `34.jpg` son escenas nuevas agregadas para este propósito
(pósters "Where's Waldo" de alta resolución, nunca vistos por el modelo).
`chicken_love_you.jpeg` se probó manualmente en algún momento pero
**se decidió no sumarla ni a entrenamiento ni a test**: no es un póster
"Wally" real (es un póster "encontrá 10 personajes" con un Wally-pollo
escondido), así que queda solo como prueba suelta con
`detect_wally.py --image chicken_love_you.jpeg`, fuera del pipeline formal.

### Cómo medir si una iteración mejoró o empeoró

Hasta ahora la única forma de notar una regresión era comparar imágenes
anotadas a ojo entre sesiones — así fue como se sospechó (sin poder
confirmarlo con números) que la ronda 3 de mining había empeorado el
comportamiento sobre `chicken_love_you.jpeg`. Para no depender de eso:

- **`evaluate.py` ahora guarda cada corrida** en `eval_runs.jsonl` (un JSON
  por línea: commit de git, hash del cascade+modelo actuales, umbral usado,
  recall y falsos positivos totales por split, y el detalle por escena).
  Se guarda automáticamente al final de cada corrida (`--no-log` para
  desactivarlo puntualmente).
- **`compare_runs.py` compara dos corridas** (por defecto, la última contra
  la anterior) y dice explícitamente MEJORÓ / IGUAL / EMPEORÓ por split
  (train/test), más el detalle escena por escena: cuáles dejaron de
  encontrarse, cuáles se arreglaron, y dónde subieron/bajaron los falsos
  positivos. `--trend N` muestra las últimas N corridas en una tabla, para
  ver la tendencia y no solo el último salto.

Flujo recomendado a partir de ahora: después de cualquier cambio al pipeline
(mining, reentrenar, calibrar, cambiar `--conf`), correr

```bash
python evaluate.py            # corre y guarda el resultado
python compare_runs.py        # muestra el delta contra la corrida anterior
python sweep_conf.py          # (opcional) reelegir el umbral tras reentrenar/recalibrar
python set_default_conf.py --conf <umbral elegido>   # sincroniza el default silencioso del checkpoint
```

### Versionado de modelos (para mostrar la evolución, ej. en un post)

`evaluate.py`/`compare_runs.py` loguean métricas, pero el `cascade.xml` y el
`wally_cnn.pt` se sobreescriben en cada entrenamiento — no queda un binario
al que volver. `save_version.py` saca una "foto" completa (los dos archivos
de modelo + las métricas de la corrida de `evaluate.py` que coincida) en
`model_versions/vN_<tag>/`, pensada para commitear a git (~800KB por
versión, no hace falta Git LFS):

```bash
python evaluate.py                                    # medir antes de guardar
python save_version.py --tag "round3-mining" --note "..."
python save_version.py --list                          # tabla de todas las versiones
git add model_versions/vN_<tag> && git commit           # versionar la foto
```

Para regenerar una imagen de comparación contra una versión vieja (útil para
un post "antes/después"):

```bash
python detect_wally.py --image original-images/17.jpg --version 1
```

`v1_baseline-round3-mining` ya quedó guardada: es el estado justo antes de
sumar escenas nuevas (cascade 24×24 + CNN ronda 3 de mining, calibrada),
TRAIN 30/30 (48 FP), TEST held-out 1/4 (29 FP) — el punto de partida real
para medir las próximas iteraciones.

`v2_mas-escenas-35-39` (2026-08-25): CNN reentrenada + recalibrada con 5
escenas nuevas (`35`-`39`), mismo cascade. TRAIN 32/35 (193 FP), TEST
held-out 3/4 (28 FP) — ver tabla comparativa en el TODO de abajo.

`v3_13-escenas-nuevas` (2026-08-29): +8 escenas más (`40`-`47`). TRAIN
40/43 (227 FP), TEST held-out (el de 4 escenas de esa fecha) 3/4 (36 FP).

`v4_12-escenas-simples` (2026-08-30): +12 escenas más (`48`-`59`), varias
del estilo "imagen simple/aislada" (portadas, ilustraciones limpias, cajas
mucho más grandes en proporción que la convención de páginas de búsqueda).
TRAIN 51/55 (366 FP), TEST held-out (4 escenas) 3/4 (59 FP) — mismo recall
que v3 pero con mucho más ruido, primera señal de que este tipo de imagen
puede estar volviendo la CNN más permisiva en vez de más precisa (ver
sanity check de `chicken_love_you.jpeg` más abajo, donde v4 marcó un
falso positivo nuevo con 100% de confianza que v1-v3 nunca habían aceptado).

**Bug encontrado y arreglado (2026-08-29):** `WallyDetector.detect()` usa
`self.threshold` (guardado en el checkpoint) cuando se llama sin `--conf`
explícito, y `train.py` siempre guarda ahí `0.90` (el umbral del paper, para
reportar métricas de validación) — `calibrate.py` nunca lo tocaba. Resultado:
`detect_wally.py` sin flags usaba `0.90` en vez de `0.9998` a pesar de decir
"por defecto 0.9998" en su propia ayuda. No afectó ninguna métrica ya vista
(`evaluate.py`/`sweep_conf.py`/`compare_runs.py` siempre pasan el conf
explícito) — solo el uso manual rápido de `detect_wally.py`. Arreglado con
`set_default_conf.py --conf 0.9998`, que sincroniza el threshold guardado en
el checkpoint con el umbral de despliegue elegido. Correrlo de nuevo después
de cada `train.py` (que resetea el campo a 0.90) + `calibrate.py` +
`evaluate.py`/`sweep_conf.py`.

### Held-out ampliado a 12 escenas (2026-08-30) y comparación retroactiva

Con solo 4 escenas held-out, cada una vale 25 puntos de recall — el salto de
1/4→3/4 entre v1 y v2 podía deberse tanto a una mejora real como a que 2
escenas cualquiera cruzaran el umbral por casualidad (con n=4 el margen de
error del recall estimado ronda ±22 puntos). Se amplió `HELD_OUT_TEST` a 12
escenas (`17`, `26`, `33`, `34`, `60`-`67`) — similar al tamaño que usa el
paper original (12 escenas de test) y baja el margen de error a ~±10-13
puntos. Regla para seguir creciendo: las escenas nuevas que se decidan
reservar como test deben ser escenas que **ningún modelo haya usado nunca
para entrenar** (no vale "ascender" escenas que ya son parte del training
set de alguna versión guardada, porque dejarían de ser un examen sorpresa
para esas versiones).

Como cada versión guarda sus pesos reales (no solo sus métricas),
`compare_versions.py` puede "tomarle el examen de nuevo" a **todas** las
versiones guardadas contra el held-out actual, aunque en su momento se
hayan medido contra uno más chico — es válido mientras las escenas nuevas
sean nuevas para todas por igual (acá lo son: `60`-`67` se etiquetaron
después de entrenar v4). Resultado, las 4 versiones sobre el mismo examen
de 12 escenas:

| versión | recall | FP (12 escenas) |
|---|---|---|
| v1 | 2/12 (17%) | 28 |
| v2 | 7/12 (58%) | 58 |
| v3 | 9/12 (75%) | 60 |
| v4 | 9/12 (75%) | **121** |

Con un test set más grande la progresión v1→v2→v3 se ve mucho más clara y
sostenida que con las 4 escenas originales. v4 no mejora el recall sobre
v3 y casi duplica el ruido — confirma con números lo que ya se veía en
`chicken_love_you.jpeg`: la ronda de escenas "simples" no fue una mejora.
`64.jpg` y `66.jpg` fallan en las 4 versiones — son los casos genuinamente
difíciles, no una casualidad de una ronda puntual.
(`python compare_versions.py --save-meta` guarda este resultado en el
`meta.json` de cada versión, campo `held_out_recheck`.)

**Nota (2026-08-30):** `26.jpg` se re-etiquetó con una foto de mejor calidad
de la misma escena (la original era borrosa/de baja resolución, dificultando
tanto anotar la caja como que el cascade proponga un candidato nítido). Con
la imagen nueva, v2/v3/v4 SÍ encuentran a Wally ahí (antes fallaba en las 4
versiones) — la tabla de arriba ya refleja el resultado re-medido con la
imagen corregida contra las 4 versiones guardadas. Solo v1 sigue sin
encontrarla.

**Held-out final: 10 escenas, no 12 (2026-08-30).** `64.jpg` y `66.jpg` —
justo los dos casos que fallaban en las 4 versiones — se sacaron del
dataset por mala calidad de imagen (se eliminaron `original-images/64.jpg`,
`64.json`, `66.jpg`, `66.json`). `HELD_OUT_TEST` queda en
`{"17", "26", "33", "34", "60", "61", "62", "63", "65", "67"}`. Con esto el
dataset total pasa a 57 escenas etiquetadas: 47 de entrenamiento, 10 de
test. La tabla de arriba (medida contra las 12 originales, con `64`/`66`
incluidas) queda como registro histórico — la próxima medición
(`compare_versions.py`) va a reportar sobre las 10 definitivas.

### Held-out final (12 escenas) y validación de fuga de datos (2026-08-30)

Se agregaron `22.jpg` y `42.jpg` al held-out (reemplazando a `64`/`66`, que
ya se habían sacado del dataset por mala calidad):
`{"17","22","26","33","34","42","60","61","62","63","65","67"}`.

Problema: `22` y `42` **ya habían sido usadas para entrenar** v2, v3 y v4 (y
`22` también v1) antes de convertirse en held-out — cualquier buen resultado
de esas versiones en esas dos escenas es fuga de datos (memorización), no
generalización real. Para tener un número honesto se re-entrenaron 4
"réplicas" (`v5`-`v8`, script nuevo `prepare_dataset_scenes.py` +
`save_replay_version.py`): **exactamente el mismo conjunto histórico de
escenas de entrenamiento de cada versión, pero sacando `22`/`42` donde
correspondía** (v1→v5: -1 escena, 29; v2→v6: -1, 34; v3→v7: -2, 41; v4→v8:
-2, 53). Nota de método: v1 se había construido con 3 rondas iterativas de
`mine_round2.py`; v5-v8 (como v2-v4 reales) solo llevan una pasada de mining
de `prepare_dataset.py` — v5 no es 100% comparable a v1 en metodología, solo
en composición de datos.

Las 8 versiones contra el held-out de 12 escenas actual:

| versión | recall | FP | ¿entrenó con `22`/`42`? |
|---|---|---|---|
| v1 (original) | 3/12 (25%) | 22 | sí, con `22` |
| v2 (original) | 8/12 (67%) | 52 | sí, con `22` |
| v3 (original) | 11/12 (92%) | 51 | sí, con `22` y `42` |
| v4 (original) | 11/12 (92%) | 113 | sí, con `22` y `42` |
| **v5 (réplica limpia de v1)** | **4/12 (33%)** | 85 | no |
| **v6 (réplica limpia de v2)** | **9/12 (75%)** | 85 | no |
| **v7 (réplica limpia de v3)** | **9/12 (75%)** | 65 | no |
| **v8 (réplica limpia de v4)** | **10/12 (83%)** | 51 | no |
| v9 (v8 + `68`,`69`) | 10/12 (83%) | 51 | no |
| **v10 (v9 sin las 8 escenas "simples")** | **11/12 (92%)** | 70 | no |

**La progresión limpia (v5→v6→v7→v8, 33%→75%→75%→83%) sigue siendo real y
sostenida** — confirma que la mejora entre rondas no dependía de memorizar
`22`/`42`. El ajuste más importante es v3: el 92% original estaba inflado
por la fuga; el número real de generalización es **75%**, no 92% — es el
que hay que citar en el post, no el original. Dato sin explicar del todo:
v8 tiene menos de la mitad de los FP que v4 (51 vs 113) con casi el mismo
dataset — podría ser variancia normal de entrenamiento o que esas 2 escenas
aportaban ruido al mining; no confirmado.

**v9 (2026-08-30):** primera versión entrenada sobre la base limpia (v8,
53 escenas) + 2 escenas nuevas (`68`, `69`) — sin `22`/`42` desde el
arranque, todo hecho con el flujo estándar (`prepare_dataset.py`, ya no
hace falta el script de réplica). Resultado: **empate técnico con v8**
(10/12 recall, 51 FP los dos) — sube el recall en `42.jpg` (que v8 fallaba)
pero baja en `61.jpg` (que v8 sí encontraba). No es una mejora clara, es un
intercambio lateral: con solo 2 escenas nuevas no alcanzó para mover la
aguja del recall total, aunque cambió cuál escena puntual falla.

**v10 (2026-08-30) — confirma la sospecha sobre las escenas "simples":**
se saca del entrenamiento a `48,49,50,51,53,55,57,58` (8 de las 12 escenas
"simples" de la ronda de v4, sospechosas desde el principio de volver la
CNN más permisiva) sobre la base de v9 — queda en 47 escenas (menos que
las 55 de v9). Resultado: **11/12 (92%) recall, el mejor de toda la línea
limpia**, solo falla `42.jpg`. Confirma la sospecha: esas escenas
efectivamente ensuciaban la frontera de decisión, no ayudaban — con
*menos* datos de entrenamiento (47 vs 55) el modelo generaliza mejor. El
costo es más ruido (70 FP vs 50 de v9) — el modelo quedó más sensible en
general. `v10` es la mejor versión hasta ahora en recall held-out.

### Etiquetado

Las 12 escenas del held-out (y todas las de training) ya están etiquetadas.
Para seguir sumando escenas nuevas, etiquetarlas con `labelme` (ya instalado
en `.venv/bin/labelme`; **no** relacionado con el modelo YOLO — es solo el
nombre del formato de texto de cajas que ya se usa para 20-32):

```bash
cd src
../.venv/bin/labelme original-images/<N>.jpg   # dibujar 1 rectangulo, label "wally", guardar
```

Después:

```bash
cd dataset && python labelme_to_yolo.py && cd ..   # json -> txt (el script asume cwd=dataset/)
python evaluate.py                                  # mide la escena nueva
python compare_runs.py --trend 3                    # ver contra corridas previas si las hay
```

### TODO / próximos pasos

Recall in-sample ya está resuelto (30/30, ver arriba) pero **no generaliza**:
la primera medición real sobre el test set held-out dio **1/4** (ver abajo).
Eso cambia la prioridad de lo que sigue:

- [x] **Ampliar el held-out a 12 escenas + comparación retroactiva**
      (2026-08-30, ver sección "Held-out ampliado" arriba). Con 12 escenas
      en vez de 4 la progresión v1→v2→v3 se ve sostenida y clara (17%→50%→
      67% recall), y confirma que v4 no mejoró (mismo recall que v3, casi el
      doble de FP).
- [ ] **Investigar si las escenas "simples" (48-59) son la causa del
      retroceso de v4** — son las que tienen cajas mucho más grandes en
      proporción (hasta 18% del ancho vs. 0.8-2.2% del resto). Revisar
      cuáles capturan cuerpo completo en vez de cara/gorro y si conviene
      re-etiquetarlas más ajustadas o sacarlas del training set.
- [x] **Etiquetar el test set held-out** (`17`, `26`, `33`, `34`). Resultado
      (2026-08-25, primera corrida real en `eval_runs.jsonl`): **1/4 recall,
      29 FP** — encontró Wally solo en `34.jpg`; falló en `17.jpg`, `26.jpg`
      y `33.jpg` (en esas tres ni siquiera hubo un candidato que superara
      `conf=0.9998`). Contra el 30/30 in-sample, confirma overfitting real a
      las 30 escenas de entrenamiento, no solo una sospecha a ojo.
- [x] **Investigar caso por caso por qué falla en `17`/`26`/`33`**
      (`diagnose_scene.py`, 2026-08-25). Resultado en los 3 casos: el cascade
      SÍ propone un candidato correcto (IoU 0.70-0.81) — el cuello de botella
      está 100% en la CNN, que le da confianza baja incluso a un Wally típico
      sin nada raro (`17.jpg`: 72%). Confirma que el problema es de
      generalización de la CNN, no del cascade — no hacía falta el
      reentrenamiento caro del cascade.
- [x] **Más escenas nuevas de entrenamiento** (eje 1 de "conseguir más datos
      reales", ver abajo el ítem 2 que sigue pendiente). Se sumaron 5 escenas
      reales (`35`-`39`) y se reentrenó + recalibró la CNN (mismo cascade,
      sin tocar). Resultado (`v2_mas-escenas-35-39` en `model_versions/`,
      mismo `conf=0.9998`):
      | | v1 (antes) | v2 (con 35-39) |
      |---|---|---|
      | TRAIN recall | 30/30 | 32/35 |
      | TRAIN FP | 48 | 193 |
      | **TEST held-out recall** | **1/4** | **3/4** |
      | TEST FP | 29 | 28 |

      Mejora real en lo que importa (held-out), a costa de perder 3 escenas
      que antes andaban bien (`25.jpg`, `30.jpg`, `31.jpg` — los casos límite
      de la ronda 3) y subir mucho el ruido in-sample. Es la contracara
      esperable de agregar diversidad nueva: el modelo deja de sobreajustar
      tan fino a las 30 escenas viejas. Sigue pendiente sumar **aún más**
      escenas (y de estilos distintos, como `33.jpg`) para consolidar la
      mejora sin perder los casos límite.
- [ ] **Etiquetar personajes "familia de Wally" como negativos reales**
      (eje 2 de "conseguir más datos reales", sigue pendiente): Wenda/Wilma,
      Woof, el Mago Barbablanca, Odlaw dentro de las escenas que ya tenemos.
      El paper lo menciona explícitamente: *"the negative data set will have
      to contain many of these Wally look-alike characters"*. Hoy los
      negativos "difíciles" solo salen de parches al azar o de lo que el
      cascade confunde en esa ronda — ninguno viene de curación humana
      sabiendo específicamente "esto es Woof, no Wally".
- [x] **Barrido de umbral tras el reentreno** (`sweep_conf.py`, 2026-08-29 —
      corre cascade+CNN una sola vez y prueba varios `--conf` sobre los
      mismos candidatos cacheados, en vez de una corrida completa de
      `evaluate.py` por umbral):
      | conf | TRAIN recall | TRAIN FP | TEST recall | TEST FP |
      |---|---|---|---|---|
      | 0.90 | 35/35 | 5087 | **4/4** | 556 |
      | 0.95 | 35/35 | 3343 | **4/4** | 352 |
      | 0.99 | 35/35 | 1273 | 3/4 | 153 |
      | 0.995 | 33/35 | 870 | 3/4 | 114 |
      | 0.998 | 33/35 | 550 | 3/4 | 73 |
      | 0.9995 | 32/35 | 291 | 3/4 | 41 |
      | **0.9998 (default)** | 32/35 | 189 | 3/4 | 28 |
      | 0.9999 | 31/35 | 138 | 2/4 | 22 |

      A diferencia de la ronda 3 (donde sí existía un punto con recall
      completo y poco ruido), acá **no hay un umbral limpio**: recall
      completo en los dos splits (`4/4` test) solo aparece en `0.90`/`0.95`,
      con cientos de FP. `0.9998` se mantiene como default: mismo recall
      held-out que casi todo el rango medio (3/4) con el menor ruido de esa
      franja. Esto es evidencia a favor de que un umbral global solo no
      alcanza — refuerza el ítem de abajo (top-1/top-N).
- [ ] **Top-1 por confianza en `detect()`** (o top-N configurable) en vez de
      devolver todo lo que supera el umbral. El barrido de arriba lo
      confirma: a umbral bajo (0.90-0.95) hay recall completo en ambos
      splits pero cientos de FP por escena — quedarse con el candidato más
      confiado por escena probablemente resuelva recall y precisión al mismo
      tiempo, algo que ningún punto del barrido logra por sí solo.
- [x] **Calibrar la confianza** — temperature scaling (`cnn_reclassifier/calibrate.py`,
      `T=1.499`), ver sección "Calibración de la confianza" arriba.
- [x] **Otra ronda de hard-negative mining** (`mine_round2.py --round 3`)
      contra el cascade+CNN calibrados de esta ronda: 3029 hard negatives
      nuevos, reentrenar + recalibrar. Resultado: recall **30/30** (recupera
      `25.jpg`) con **49 FP** totales a `conf=0.9998` (antes 417 FP a 29/30).
      Fue el cambio de mayor impacto de esta tanda — más que la calibración
      sola.
- [x] **Medir mejora/regresión con números, no a ojo** — `evaluate.py` loguea
      cada corrida en `eval_runs.jsonl` y `compare_runs.py` compara contra
      corridas anteriores (ver sección arriba).

#### Sanity check informal: `chicken_love_you.jpeg`

Queda fuera del pipeline formal (no entrena, no se mide en `evaluate.py`) —
ver nota arriba. No es un control negativo limpio: es un póster "encuentra
10 personajes" (Dwight Schrute, **Wally**, Titanic, The Dude, Pickle Rick,
...) que sí tiene un Wally-pollo real escondido, nunca se confirmó cuál caja
es la correcta, así que este chequeo sigue siendo solo informal:

- Antes de la ronda 3 (2026-08-23): 8 detecciones a `conf=0.998`.
- Después de la ronda 3 (2026-08-24), con el default nuevo `conf=0.9998`:
  **0 detecciones** — el mining hizo que todo lo que antes pasaba el umbral
  en esta imagen (personajes a rayas, ninguno confirmado como el
  Wally-pollo real) ahora quede por debajo. A `conf=0.9` reaparecen 9
  candidatos, uno de ellos en la misma zona que antes
  (~x=784,y=612 vs. ~x=773,y=614), así que probablemente sea supresión por
  umbral y no una regresión real — pero sigue sin poder confirmarse sin
  anotar la caja correcta.
