import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=20):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image,lines):
    
    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]

    left_lines = [] 
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:

            if x1 == x2:
                pass 
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m,c))
                elif m >= 0:
                    right_lines.append((m,c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)


    for slope, intercept in [left_line, right_line]:

        rows, cols = image.shape[:2]
        y1= int(rows) 

        y2= int(rows*0.6) 
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1,y1,x2,y2]]]))
    
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts = np.array([poly_vertices],'int32'), color = (0,255,0))
    return cv2.addWeighted(image,0.7,img,0.4,0.)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = slope_lines(line_img,lines)
    return line_img


def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):

    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)

    return lines_edges

def get_vertices(image):

    # Para video propio
    rows, cols = image.shape[:2]
    bottom_left  = [0, rows-140]
    top_left     = [470, 290]
    bottom_right = [720, 290]
    top_right    = [cols, rows-140] 

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver


def lane_finding_pipeline(image):
    
    #Grayscale
    gray_img = grayscale(image)
    #Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 1)
    #Canny Edge Detection
    canny_img = canny(img = smoothed_img, low_threshold = 110, high_threshold = 240)
    #Masked 
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image))
    #Hough Transform Lines
    houghed_lines = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    #Draw lines on edges
    output = weighted_img(img = houghed_lines, initial_img = image, α=0.8, β=1., γ=0.)
    
    return output



video_path = 'Data_Set3.mp4'
#video_path = 'IMG_0310.MOV'
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Resultado', lane_finding_pipeline(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

