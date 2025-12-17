from ultralytics import YOLO

# Ruta a tu modelo entrenado
MODEL_PATH = "runs/detect/train7/weights/best.pt"

# Imagen o carpeta a evaluar
SOURCE = "/Users/danielvargas/Documents/wally/src/1.jpg"
#SOURCE = "/Users/danielvargas/Downloads/WhatsApp Image 2025-10-23 at 01.33.17.jpeg"
SOURCE = "/Users/danielvargas/Documents/wally/src/original-images/2.jpg"
# También puede ser una carpeta, por ejemplo:
# SOURCE = "dataset/images/val"

# Cargar el modelo
model = YOLO(MODEL_PATH)

# Ejecutar inferencia
results = model.predict(
    source=SOURCE,
    conf=0.1,     # umbral de confianza (ajustable)
    save=True      # guarda las imágenes con detecciones
)

print("Inferencia completada ✅")
