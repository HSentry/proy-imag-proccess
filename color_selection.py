import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


#image = mpimg.imread('Img_test\im1.jpg')
image = mpimg.imread('Img_test\im2.jpg')

ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Color 
red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]



plt.subplot(2,1,1)
plt.imshow(image)
plt.title("Imagen")
plt.subplot(2,1,2)
plt.imshow(color_select)
plt.title("Color seleccionado")
plt.show()

