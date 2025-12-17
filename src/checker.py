import struct
import numpy as np
import matplotlib.pyplot as plt

def read_vec_file(vec_file, force_w=None, force_h=None):
    with open(vec_file, 'rb') as f:
        num = struct.unpack('<i', f.read(4))[0]
        size = struct.unpack('<i', f.read(4))[0]
        width = struct.unpack('<h', f.read(2))[0]
        height = struct.unpack('<h', f.read(2))[0]

        # Si no se guardaron correctamente, usamos los forzados
        if width == 0 or height == 0:
            width, height = force_w, force_h

        print(f"Samples: {num}, Size per sample: {size}, Width: {width}, Height: {height}")

        images = []
        for _ in range(num):
            data = f.read(size)
            if len(data) != size:
                print("⚠️  Warning: unexpected end of file.")
                break
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            images.append(img)
        return images

# ⚠️ fuerza manualmente el tamaño que sabes que usaste
images = read_vec_file("positives.vec", force_w=64, force_h=64)

# Mostrar las primeras muestras
ncols = 5
nrows = (len(images) + ncols - 1) // ncols
plt.figure(figsize=(12, 2.5 * nrows))
for i, img in enumerate(images[:ncols*nrows]):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Sample {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

