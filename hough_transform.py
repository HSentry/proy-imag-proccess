import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('Img_test\im2.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define kernel 
# Kernel para imagen propia (im1)
kernel_size = 1

#Kernel para imagen dataset (im2)
#kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


# Parametros Canny para imagen propia (im1)
low_threshold = 110
high_threshold = 240
# Parametros Canny para imagen dataset (im2)
#low_threshold = 180
#high_threshold = 240
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# Poligono para imagen propia (im2)
imshape = image.shape
vertices = np.array([[(0,imshape[0]-140),(470, 290), (720, 290), (imshape[1],imshape[0]-140)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)


#Poligono para imagen dataset (im1)
# imshape = image.shape
# vertices = np.array([[(170,imshape[0]-120),(500, 420), (700, 420), (imshape[1]-170,imshape[0]-120)]], dtype=np.int32)
# cv2.fillPoly(mask, vertices, ignore_mask_color)
# masked_edges = cv2.bitwise_and(edges, mask)


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 2     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 4 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
lines_edges = cv2.polylines(lines_edges,vertices, True, (0,0,255), 10)

plt.subplot(2,1,1)
plt.imshow(image)
plt.title("Imagen")
plt.subplot(2,1,2)
plt.imshow(lines_edges)
plt.title("Linea coloreada [Rojo] - Región de interés [Azul]")
plt.show()

