import numpy as np
from PIL import Image


# LM crop
with open("./LM_crop.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (1881, 926), image_data, "raw")
    image.save("./LM_crop_.jpg")


with open("./LM_resized.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./LM_resized.png")

with open("./LM_resize_original.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./LM_resize_original.png")




#
with open("./car1_resized.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./car1_resized.png")

with open("./car1_resized_original.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./car1_resized_original.png")

with open("./pixel_resized.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (16, 16), image_data, "raw")
    image.save("./pixel_resized.png")

with open("./pixel_resized_original.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (16, 16), image_data, "raw")
    image.save("./pixel_resized_original.png")


# with open("./car2_resized.raw", "rb") as f:
#     image_data = f.read()
#     image = Image.frombytes("L", (572,342), image_data, "raw")
#     image.save("./car2_resized.jpg")


# with open("./car3_resized.raw", "rb") as f:
#     image_data = f.read()
#     image = Image.frombytes("L", (228, 174), image_data, "raw")
#     image.save("./car3_resized.jpg")

