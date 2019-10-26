# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:55:25 2019

@author: Mahmoud Nada
"""

import cv2
import numpy as np
from keras.models import load_model
import os

# First we import our labels from the dataset diectory
Ulabels = sorted(os.listdir('./asl_alphabet_train'))
print(Ulabels)

# we must create a dictionary of labels and their corresponding numeric value to
# interpret the prediction of the model
labels_dict = {i:l for i,l in enumerate(Ulabels)}
print(labels_dict)

# Their are 2 final models , one was trained from scratch (Local2) and the other
# one was a pre-trained model (VGG16)
local_model = './Local2.h5'
VGG16_model = './VGG16.h5'

# we can try both of the models each at a time.
model = load_model(local_model)

# the code below opens the WebCam and draws a region on the lift of the screen 
# this region is where you supposed to opsition your palm
# this region is then resized into 64x64 pixels then randomized and fed to the model
# our model immediatly predicts the input frame and returnes a numeric form of prediction
# it then interpreted to the original label by the dictionary we built above 
# the predicted label is then put into the top left of screen.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    cv2.rectangle(frame, (20,100), (250,350),(255,0,0), 2)
    
    hand = frame[100:350, 20:250]
    
    image = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    image = cv2.resize(hand, (64, 64))
    image = image.astype("float") / 255.0
    
    pred = model.predict(np.expand_dims(image, axis=0))
    
    try:
        #cv2.putText(frame, labels_dict[np.argmax(pred, axis=1)[0]], (250, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(hand, labels_dict[np.argmax(pred, axis=1)[0]], (50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    except KeyError:
        cv2.putText(frame, "Not Defined", (250, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(hand, "Not Defined", (250, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.imshow('ROI View', hand)    
    cv2.imshow('Full View', frame)
    
    
    
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()








