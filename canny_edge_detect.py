import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


image = mpimg.imread('Img_test\im2.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Gaussian smoothing / blurring
kernel_size = 3 
#kernel_size = 1 
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and run it
# low_threshold = 180
# high_threshold = 240
low_threshold = 110
high_threshold = 240
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


plt.subplot(2,1,1)
plt.imshow(image)
plt.title("Imagen")
plt.subplot(2,1,2)
plt.imshow(edges, cmap='Greys_r')
plt.title("Canny Edge")
plt.show()