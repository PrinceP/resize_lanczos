import numpy as np
from PIL import Image, ImageOps

for path in ["LM_crop.jpg", "car1.jpg","car2.jpg","car3.jpg", "pixel.jpg"]:


    im = Image.open(path)

    # data = np.array(im)
    # RGB
    # flattened = data.flatten()

    # RRRGGGBBB
    # flattened = data.T.flatten()

    # Grayscale
    im_gray = ImageOps.grayscale(im)

    data_rgb = np.array(im)
    data_gray = np.array(im_gray)

    data_rgbflat = data_rgb.flatten()
    data_grayflat = data_gray.flatten()
    with open(path.replace('.jpg','.raw'), "wb") as f:
        for i in data_grayflat:
            f.write(i)



