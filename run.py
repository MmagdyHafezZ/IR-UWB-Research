import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_array = np.load('raw_data.npy')

plt.imshow(img_array, cmap='gray') 
plt.show()

if img_array.dtype != np.uint8:
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    img_array = img_array.astype(np.uint8)
im = Image.fromarray(img_array)
im.show()