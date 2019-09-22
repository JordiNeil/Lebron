import cv2, PIL
import numpy as np
import time
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import colorsys

cap = cv2.VideoCapture(6)
green_bajo = np.uint8([[[107,142,65]]])
green_alto = np.uint8([[[107+10,142+10,65+10]]])
hsv_green_bajo = cv2.cvtColor(green_bajo,cv2.COLOR_BGR2HSV)
hsv_green_alto = cv2.cvtColor(green_alto,cv2.COLOR_BGR2HSV)
print(hsv_green_bajo)
print(hsv_green_alto)

while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    verdes_bajos = np.array([78,70,70]) #verdes_bajos = np.array([49,50,50])
    verdes_altos = np.array([93,200,200]) #verdes_altos = np.array([107,255,255])
    mask = cv2.inRange(hsv, verdes_bajos, verdes_altos)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mas = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150 and area < 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            momentos = cv2.moments(contour)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            #Dibujar el centro
            cv2.circle(frame,(cx, cy), 3, (0,0,255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    plt.figure()
    print(ids)
    plt.imshow(frame_markers)
    
    for i in range(len(ids)):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
    plt.legend()
    plt.show()
    
    key = cv2.waitKey(1)
    if key == 27:
        break
            
cap.release()
cv2.destroyAllWindows()
