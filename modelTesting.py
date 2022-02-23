import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
from modules.gestureMath import *

import numpy as np
import csv
import pandas as pd
import pickle

from math import *
import json 
import random

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()

    result = dict()
    result[0]='Doctor Strange'
    result[1]='Hi'
    result[2]='Stop'
    result[3]='Victory'
    result[100] = 'None'
    columnLimit = 20
    
    loadedModel = pickle.load(open('RFCModel.sav','rb'))

    selectedHandPoints = [0,4,8,20]
    handPointSize = len(selectedHandPoints)
    lm = [[0 for i in range(handPointSize)] for j in range(2)]
    dif = [[0 for i in range(handPointSize)] for j in range(2)]

    counter = 0
    testList = []
    answer = 100
    prevlmlist = [[0,0,0] for i in range(21)]

    while True:
        try:
            success,img = cap.read()
            img = detector.findhands(img)
            lmlist = detector.findPosition(img)
        except:
            continue
        
        cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
        cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
        cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

        if counter % columnLimit == 0 and len(testList) != 0:
            answer = loadedModel.predict([testList])
            testList = []
            counter = 0

        if len(lmlist):

            x_list = [i[1] for i in lmlist]
            y_list = [i[2] for i in lmlist]

            origin = (min(x_list), min(y_list))
            terminal = (max(x_list), max(y_list))
            boxLength = terminal[0] - origin[0]
            boxHeight = terminal[1] - origin[1]
            boxDiagonal = sqrt(boxLength*boxLength + boxHeight*boxHeight)
            center = ((int)(origin[0]+boxLength/2), (int)(origin[1]+boxHeight/2))

            cv2.rectangle(img, origin, terminal, color=(0,0,255), thickness=2)
            cv2.circle(img, origin, 3, (255,0,0), cv2.FILLED)
            cv2.circle(img, terminal, 3, (255,0,0), cv2.FILLED)
            cv2.circle(img, center, 5, (0,255,0), cv2.FILLED)

            testList.append(boxLength / boxHeight)
            for i in selectedHandPoints:
                cv2.arrowedLine(img, center, (lmlist[i][1], lmlist[i][2]), (0,0,0), 2)
                cv2.arrowedLine(img, (prevlmlist[i][1], prevlmlist[i][2]), (lmlist[i][1], lmlist[i][2]), (255,255,255), 2)

                dist, angle = getVector(center,(lmlist[i][1], lmlist[i][2]))
                distC, angleC = getVector((prevlmlist[i][1], prevlmlist[i][2]),(lmlist[i][1], lmlist[i][2]))

                testList.append(dist/boxDiagonal)
                testList.append(angle)
                testList.append(distC/boxDiagonal)
                testList.append(angleC)

            counter = (counter+1) % columnLimit
            cv2.putText(img, result[int(answer)], (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            prevlmlist = lmlist

        else:
            cv2.putText(img, " (No Hands Detected)", (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        
        cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
    
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break


if __name__=="__main__":
    main()