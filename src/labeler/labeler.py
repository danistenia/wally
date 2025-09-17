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

    def build_patches(self, square_area: int):
        #print(self.img.x_length)
        #print(self.img.y_length / 64)

        #iterator_y = count(start=0, step=square_area)
        #iterator_x = count(start=0, step=square_area)

        #print(self.img.x_length)
        x_stopper = round(self.img.x_length / square_area)
        y_stopper = round(self.img.y_length / square_area)

        #print(y_stopper)

        for x in range(0, self.img.x_length, 64):
            #print(y)
            for y in range(64, self.img.y_length, 64):
                print(y, x)

    def building_coordanates(self, square_area):

        x_stopper = round(self.img.x_length / square_area)
        y_stopper = round(self.img.y_length / square_area)

        self.patch_dict = {
            f"row_{k + 1}": 
            {
                f"col_{jdx + 1}": None for jdx in range(0, round(self.img.x_length / square_area))
            } for k in 
            range(0, round(self.img.y_length / square_area))
        }

        #print(self.patch_dict)



        for idx, (y, y_prima) in enumerate(zip(range(0, self.img.y_length, square_area), 
                              range(square_area, self.img.y_length, square_area))):
            

            for jdx, (x, x_prima) in enumerate(zip(range(0, self.img.x_length, square_area), 
                              range(square_area, self.img.x_length, square_area))):
            
                #print(f"y: {y, y_prima} y x: {x, x_prima}")

                # Save dictionaries key, value pairs.
                self.patch_dict[f"row_{idx + 1}"][f"col_{jdx + 1}"] = {
                                                        "y": (y, y_prima), 
                                                        "x": (x, x_prima)
                                                     }



            #print(f"{'_' * 30}")

        


        





if __name__ == "__main__":
    
    img_path = "/Users/danielvargas/Documents/wally/original-images/1.jpg"

    image = ImageInput(img_path=img_path)
    pb = PatchBuilder(img=image)

    # print("Size de la imagen:", image.size)

    #print(2048/64)
    #print(pb.build_patches(square_area=10))

    #pb.build_patches(square_area=64)

    pb.building_coordanates(square_area=64)
    #print("Size de la imagen:", image.size)

    #print(json.dumps(pb.patch_dict, indent=4))
    print(pb.patch_dict)