import cv2
import numpy as np

cap = cv2.VideoCapture('ruta_del_video.mp4')

canny_threshold1 = 50
canny_threshold2 = 150
hough_rho = 1
hough_theta = np.pi/180
hough_threshold = 15
hough_min_line_length = 40
hough_max_line_gap = 20

def detect_lines(frame):
  # Aplicamos la detección de bordes de Canny
  edges = cv2.Canny(frame, canny_threshold1, canny_threshold2)

  # Aplicamos la transformada de Hough para detectar las líneas
  lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold,
                          minLineLength=hough_min_line_length,
                          maxLineGap=hough_max_line_gap)

  # Dibujamos las líneas detectadas en el marco original
  line_image = np.zeros_like(frame)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

  # Combinamos las líneas dibujadas con el marco original
  combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

  return combined_image



while True:
  # Lee un cuadro del video
  ret, frame = cap.read()

  # Salir del bucle si no hay más cuadros para leer
  if not ret:
    break

  # Aplica la detección de líneas al cuadro actual
  line_image = detect_lines(frame)

  # Muestra el resultado en una ventana
  cv2.imshow('Líneas de calle detectadas', line_image)

  # Espera una tecla para salir del bucle
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()