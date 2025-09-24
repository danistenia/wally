from dataclasses import dataclass, field
import cv2 
import numpy as np
from itertools import count, islice
import json

@dataclass
class ImageInput:
    img_path: str
    img: np.ndarray = field(init=False)
    x_length: int = field(init=False)
    y_length: int = field(init=False)

    def __post_init__(self):
        self.img = cv2.imread(self.img_path)
        self.x_length = self.img.shape[1]
        self.y_length = self.img.shape[0]
        self.size = self.img.shape


@dataclass
class PatchBuilder:
    img: ImageInput
    patch_dict: dict[str, str] = field(init=False)
    square_area: int = 64

    
    def build_dict(self):

        x_stopper = round(self.img.x_length / self.square_area)
        y_stopper = round(self.img.y_length / self.square_area)

        self.patch_dict = {
            f"row_{k + 1}": 
            {
                f"col_{jdx + 1}": None for jdx in range(0, round(x_stopper))
            } for k in 
            range(0, round(y_stopper))
        }

        return self.patch_dict
        

    def build_patches(self):

        self.patch_dict = self.build_dict()

        for idx, (y, y_prima) in enumerate(zip(range(0, self.img.y_length + self.square_area, self.square_area), 
                              range(self.square_area, self.img.y_length + self.square_area, self.square_area))):
            

            for jdx, (x, x_prima) in enumerate(zip(range(0, self.img.x_length + self.square_area, self.square_area), 
                              range(self.square_area, self.img.x_length + self.square_area, self.square_area))):

                # Save dictionaries key, value pairs.
                self.patch_dict[f"row_{idx + 1}"][f"col_{jdx + 1}"] = {
                                                        "y": (y, y_prima), 
                                                        "x": (x, x_prima)
                                                     }
                
        return self.patch_dict
                
    def draw_squares(self):

        patches = self.build_patches()
        
        for row in patches.keys():
            for col in patches[row]:
                cv2.rectangle(self.img.img, 
                              (patches[row][col]["x"][0], patches[row][col]["y"][0]),
                              (patches[row][col]["x"][1], patches[row][col]["y"][1]),
                              (255, 0, 0),
                              4)
                
        cv2.imshow('Image', self.img.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_cutter(self):
        pass


if __name__ == "__main__":
    
    img_path = "/Users/danielvargas/Documents/wally/original-images/1.jpg"

    image = ImageInput(img_path=img_path)
    pb = PatchBuilder(img=image, square_area=192)

    pb.draw_squares()