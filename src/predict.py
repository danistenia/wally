import cv2
import matplotlib.pyplot as plt


cascade = cv2.CascadeClassifier('data_cascade_HAAR/cascade.xml')
image_path = './original-images/17.jpg'
image_path = '/Users/danielvargas/Downloads/WhatsApp Image 2025-10-23 at 01.33.17.jpeg'
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# for (x,y,w,h) in objects:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# cv2.imshow('result', img)
# cv2.waitKey(0)

# Carga el modelo entrenado

# Carga la imagen donde querés ver las detecciones
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecta objetos
objects = cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=0,
    minSize=(64, 64),
    maxSize=(128, 128)
)

# Si no hay detecciones
if len(objects) == 0:
    print("⚠️ No se detectaron objetos en la imagen.")
else:
    # Muestra los recortes individuales
    ncols = 5
    nrows = (len(objects) + ncols - 1) // ncols
    plt.figure(figsize=(15, 3 * nrows))

    for i, (x, y, w, h) in enumerate(objects):
        crop = img[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(crop_rgb)
        plt.title(f"Detección {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
