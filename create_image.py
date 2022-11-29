import numpy as np
from PIL import Image


# LM crop
with open("./LM_crop.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (1881, 926), image_data, "raw")
    image.save("./LM_crop_.jpg")


with open("./LM_resize.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./LM_resize.png")

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

# car1.jpg JPEG 350x174 350x174+0+0 8-bit sRGB 19.7KB 0.000u 0:00.000
# car2.jpg[1] JPEG 572x342 572x342+0+0 8-bit sRGB 45.8KB 0.000u 0:00.000
# car3.jpg[2] JPEG 228x174 228x174+0+0 8-bit sRGB 15.2KB 0.000u 0:00.000

with open("./LM_resize.raw", "rb") as f:
    image_data = f.read()
    image = Image.frombytes("L", (32, 32), image_data, "raw")
    image.save("./LM_crop_resized_testing.jpg")
