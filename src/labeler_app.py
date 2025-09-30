import argparse
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from labeler.labeler import ImageInput, PatchBuilder

# ---------- Argumentos de línea de comando ----------
parser = argparse.ArgumentParser(description="Labeler de Wally con Tkinter")
parser.add_argument("--img", required=True, help="Ruta de la imagen a etiquetar")
parser.add_argument("--size", type=int, default=64, help="Tamaño del patch cuadrado")
parser.add_argument(
    "--out", default="info.dat", help="Archivo de salida para anotaciones"
)
parser.add_argument(
    "--display-size",
    type=int,
    default=192,
    help="Tamaño de visualización del patch en la app",
)
args = parser.parse_args()

img_path = args.img
square_area = args.size
output_file = args.out
patch_display_size = args.display_size
# -----------------------------------------------------

# Cargar imagen base
imagen_base = Image.open(img_path)

# Crear patches con tu clase
image = ImageInput(img_path=img_path)
pb = PatchBuilder(img=image, square_area=square_area)
patches = pb.build_patches()

# Convertimos a lista lineal de (nombre, (x1, y1, x2, y2))
patch_list = []
for row, cols in patches.items():
    for col, coords in cols.items():
        nombre = f"{row}_{col}"
        x1, x2 = coords["x"]
        y1, y2 = coords["y"]
        patch_list.append((nombre, (x1, y1, x2, y2)))

indice = 0
bboxes = []  # Acumulador de bounding boxes (x, y, w, h)


def mostrar_patch():
    """Muestra el patch actual y actualiza el contador."""
    global indice, patch_label, patch_tk
    nombre, (x1, y1, x2, y2) = patch_list[indice]
    patch = imagen_base.crop((x1, y1, x2, y2))
    patch = patch.resize((patch_display_size, patch_display_size))
    patch_tk = ImageTk.PhotoImage(patch)
    patch_label.config(image=patch_tk, text=nombre, compound="top")
    contador_label.config(
        text=f"Patch {indice+1} de {len(patch_list)}",
        fg="black",
        font=("Arial", 12, "bold"),
    )


def finalizar():
    """Escribe una sola línea en info.dat (si hubo Wallys) y cierra la app."""
    if bboxes:
        partes = [img_path, str(len(bboxes))]
        for x, y, w, h in bboxes:
            partes += [str(x), str(y), str(w), str(h)]
        linea = " ".join(partes) + "\n"
        with open(output_file, "a") as f:
            f.write(linea)
    ventana.destroy()


def siguiente():
    """Avanza o finaliza si no hay más patches."""
    global indice
    if indice < len(patch_list) - 1:
        indice += 1
        mostrar_patch()
    else:
        finalizar()


def not_wally():
    """No se agrega bbox; solo avanzar."""
    siguiente()


def wally():
    """Marca este patch como Wally y guarda bbox absoluto."""
    global indice
    _, (x1, y1, x2, y2) = patch_list[indice]
    w = x2 - x1
    h = y2 - y1
    bboxes.append((x1, y1, w, h))
    siguiente()


# ------- Tkinter GUI -------
ventana = tk.Tk()
ventana.title("Labeler de Wally")

# Centrar ventana
ancho_ventana = 400
alto_ventana = 460
ancho_pantalla = ventana.winfo_screenwidth()
alto_pantalla = ventana.winfo_screenheight()
x = (ancho_pantalla // 2) - (ancho_ventana // 2)
y = (alto_pantalla // 2) - (alto_ventana // 2)
ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

patch_label = Label(ventana, bg="white")
patch_label.pack(pady=8)

contador_label = Label(ventana, text="", font=("Arial", 12, "bold"))
contador_label.pack(pady=6)

botonera = tk.Frame(ventana, bg="lightgray")
botonera.pack(pady=12)

boton_not_wally = Button(
    botonera,
    text="❌ Not Wally",
    command=not_wally,
    bg="#f28b82",
    fg="black",
    font=("Arial", 12, "bold"),
    width=14,
)
boton_not_wally.pack(side="left", padx=10)

boton_wally = Button(
    botonera,
    text="✅ Wally",
    command=wally,
    bg="#81c995",
    fg="black",
    font=("Arial", 12, "bold"),
    width=14,
)
boton_wally.pack(side="right", padx=10)

# Atajos de teclado
ventana.bind("<Left>", lambda e: not_wally())
ventana.bind("<Right>", lambda e: wally())
ventana.bind("<space>", lambda e: wally())

ventana.protocol("WM_DELETE_WINDOW", finalizar)

mostrar_patch()
ventana.mainloop()
