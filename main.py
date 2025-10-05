# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:56:23 2025

@author: Görkem
"""

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') #yolo model 

camera = cv2.VideoCapture(0) # open camera

while True:

    ret, frame = camera.read() # read camera 
    
    if ret:
        results = model.predict(frame, verbose=False) # Predict 

        annotated_frame = results[0].plot() # take a square object
        
        cv2.imshow("YOLOv8 Canli Tespit", annotated_frame) #show windows frame
        
        
        # Wait for press Q if press q, quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    else:
        break
camera.release()
cv2.destroyAllWindows()

