import cv2
from labeler.labeler import ImageInput
from collections import namedtuple


def find_wally_by_coords(image_path: str, x: int, y: int, w: int, h: int) -> None:
    """
    Helper para encontrar a Wally luego que ya fue etiquetado.
    Deberías ocupar tu info.dat y con esas coordenadas puedes ver si etiquetaste bien o no.

    Args:
        x: Coordenada inicial de x.
        y: Coordenada inicial de y.
        w: Es el ancho que normalmente en imágenes está referenciado al eje x.
        y: Es el alto que normalmente en imagenes está referenciado al eje y.

    Returns:
        Dibuja un rectángulo en una imagen que en teoría es Wally etiquetado.
    """

    img_obj = ImageInput(img_path=image_path)

    cv2.rectangle(img_obj.img, (x, y), (x + w, y + h), (255, 0, 0), 4)

    cv2.imshow("Wally", img_obj.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_wally_by_row_number(file_path: str, row_number: int) -> None:
    """ "
    Función para etiquetar a wally, pero por row number en el archivo info.dat.

    Args:
        file_path: Ruta del archivo info.dat.
        row_number: Número de la línea que quieres chequiar.

    Returns:
        Dibuja un rectángulo donde se supone está Wally.
    """

    with open(file=file_path, mode="r") as file:

        lines = file.readlines()
        if row_number > len(lines):
            raise ValueError(
                f"El número que indicaste en row_number: {row_number} es mayor que el número de lineas en el archivo."
            )
        else:
            line = lines[row_number - 1]
            line_split = line.split()

            ImagePatches = namedtuple(
                "ImagePatches", ["path", "wally_index", "x", "y", "w", "h"]
            )
            img = ImagePatches(
                line_split[0],
                int(line_split[1]),
                int(line_split[2]),
                int(line_split[3]),
                int(line_split[4]),
                int(line_split[5]),
            )

            img_obj = ImageInput(img_path=img.path)

            # cv2.imshow('Wally', img_obj.img)

            # print(type(img_obj.img))
            # print(img.x)
            # print(img.y)
            # print(img.w)
            # print(img.h)

            file.close()
            cv2.rectangle(
                img_obj.img,
                (img.x, img.y),
                (img.x + img.w, img.y + img.h),
                (255, 0, 0),
                4,
            )

            cv2.imshow("Wally", img_obj.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":

    img_path = "/Users/danielvargas/Documents/wally/original-images/1.jpg"

    # Example
    # find_wally_by_coords(image_path=img_path, x=512, y=512, w=256, h=256)
    find_wally_by_row_number(file_path="info.dat", row_number=1)
