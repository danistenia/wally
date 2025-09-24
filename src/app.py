"""Tkinter app for labeling Wally Images by Dani"""

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Image Display")

image_path = "/Users/danielvargas/Documents/wally/original-images/1.jpg"
#image_path = "/Users/danielvargas/Documents/Captura de pantalla 2025-09-10 a la(s) 1.10.12 a.m..png"
pil_image = Image.open(image_path)
pil_image = pil_image.resize((600, 400), Image.Resampling.LANCZOS)

tk_image = ImageTk.PhotoImage(image=pil_image)

label = tk.Label(root, image=tk_image)
label.image = tk_image  # Keep a reference
label.pack(expand=1, fill="both")


root.mainloop()