import cv2
import numpy as np

# Ruta del video
video_path = 'IMG_0264.mp4'

# Carga del video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica el filtro Gaussiano para suavizar la imagen
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta los bordes utilizando el operador de Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Define los parámetros de la transformada de Hough
    rho = 1  # Resolución de la distancia en píxeles
    theta = np.pi / 180  # Resolución del ángulo en radianes
    threshold_hough = 100  # Umbral mínimo de votos
    min_line_length = 100  # Longitud mínima de línea aceptada
    max_line_gap = 10  # Máxima brecha entre segmentos para conectarlos

    # Aplica la transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(edges, rho, theta, threshold_hough, np.array([]), min_line_length, max_line_gap)

    # Dibuja las líneas detectadas en el frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Muestra el resultado
    imS = cv2.resize(frame, (1080, 720))   
    cv2.imshow('Resultado', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()