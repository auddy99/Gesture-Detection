import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
from modules.gestureMath import *

import numpy as np
import csv
import pandas as pd

from math import *
import json 
import random

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    countLabel = 0
    p = dict()
    
    # User Inputs (Frames taken is columnLimit * sampleSpace)
    targetLabel = "Doctor Strange"
    sampleSpace = 50
    selectedHandPoints = [0,4,8,20]
    # ----------------------

    p['index'] = [targetLabel+"_" + str(i) for i in range (sampleSpace)]
    column_names = []
    column_names.append("size_ratio")
    for j in selectedHandPoints:
        column_names.append("dist_"+str(j))
        column_names.append("angle_"+str(j))
        column_names.append("distC_"+str(j))
        column_names.append("angleC_"+str(j))
    data = [[0.0 for j in range(len(column_names))] for i in range(sampleSpace)]
    df = pd.DataFrame(data, columns=column_names)
    handPointCount = len(selectedHandPoints)
    prevlmlist = [[0,0,0] for i in range(21)]

    while countLabel < sampleSpace:

        success,img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findPosition(img)

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

            df["size_ratio"][countLabel] = boxLength / boxHeight
            for i in selectedHandPoints:
                cv2.arrowedLine(img, center, (lmlist[i][1], lmlist[i][2]), (0,0,0), 2)
                cv2.arrowedLine(img, (prevlmlist[i][1], prevlmlist[i][2]), (lmlist[i][1], lmlist[i][2]), (255,255,255), 2)
                dist, angle = getVector(center,(lmlist[i][1], lmlist[i][2]))
                distC, angleC = getVector((prevlmlist[i][1], prevlmlist[i][2]),(lmlist[i][1], lmlist[i][2]))
                df["dist_"+str(i)][countLabel] = dist/boxDiagonal
                df["angle_"+str(i)][countLabel] = angle
                df["distC_"+str(i)][countLabel] = distC/boxDiagonal
                df["angleC_"+str(i)][countLabel] = angleC
            prevlmlist = lmlist
            countLabel += 1

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
        cv2.putText(img, "Frames taken: "+str(countLabel), (310,30), cv2.FONT_HERSHEY_PLAIN, 2, (150,0,0), 2)
        cv2.imshow('image1',img)

        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break

    df.insert(0,"Label", [targetLabel for i in range(sampleSpace)])
    print(df)
    df.to_feather('trainingData\\'+targetLabel+'_train.feather')

if __name__=="__main__":
    main()