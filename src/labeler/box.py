import cv2

img_path = "/Users/danielvargas/Documents/wally/original-images/1.jpg"

# image = cv2.imread(img_path)
# y_shape = image.shape[1]

# x, y = (0, 0)
# w, h = (64, 64)


# cv2.rectangle(image, (x, y), (x + w, x + h), (255, 0, 0), 4)
# #cv2.rectangle(image, (x, y), (x + w, x + h), (255, 0, 0), 4)

# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def square_iteration(img_path: str, square_area: int, color_intensity: int = 4):
    """
    Iteración sobre el x con área determinada.

    Args:
        Input de la imagen que quieres recorrer y dibujar.
    
    Returns:
        None. Solo dibuja los cuadrados en la imagen.
    """

    x, y = (0, 0)
    w, h = (x + square_area, y + square_area)
    color = (255, 0, 0)

    # Img attrs
    img = cv2.imread(img_path)
    x_length = img.shape[1]
    print("Largo total", x_length)

    while x < x_length:
        cv2.rectangle(img, (x, y), (w, h), color, color_intensity)
        x += square_area
        #print(x)

        if x >= x_length:
            #print(x_length)
            cv2.rectangle(img, (x_length - square_area, y), (x_length, y + square_area), color, color_intensity)
            
    
    # Last square
#    cv2.rectangle(img, (x_length, y), )

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#square_iteration(img_path=img_path, square_area=64)

img = cv2.imread(img_path)
cropped_img = img[0:64, 0:64]

cv2.imshow("Imagen1", cropped_img)
cv2.imshow("Imagen Original", img)
cv2.waitKey()
cv2.destroyAllWindows()

