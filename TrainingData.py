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
    
    # User Inputs (Frames taken is columnLimit * rowCount)
    columnLimit = 20
    targetLabel = ""
    rowCount = 50
    selectedHandPoints = [0,4,8,20]
    # ----------------------

    p['index'] = [targetLabel+"_" + str(i) for i in range (rowCount)]
    column_names = []
    for i in range(columnLimit):
        column_names.append("size_ratio_"+str(i))
        for j in selectedHandPoints:
            column_names.append("dist_"+str(j)+"_"+str(i))
            column_names.append("angle_"+str(j)+"_"+str(i))
    data = [[0.0 for j in range(len(column_names))] for i in range(rowCount)]
    df = pd.DataFrame(data, columns=column_names)
    columnCounter = 0
    rowCounter = 0
    handPointCount = len(selectedHandPoints)

    while countLabel < rowCount * columnLimit:

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
            cv2.rectangle(img, origin, terminal, color=(0,0,255), thickness=2)
            cv2.circle(img, origin, 3, (255,0,0), cv2.FILLED)
            cv2.circle(img, terminal, 3, (255,0,0), cv2.FILLED)

            df["size_ratio_"+str(columnCounter)][rowCounter] = boxLength / boxHeight
            for i in selectedHandPoints:
                dist, angle = getVector(origin,(lmlist[i][1], lmlist[i][2]))
                df["dist_"+str(i)+"_"+str(columnCounter)][rowCounter] = dist/boxDiagonal
                df["angle_"+str(i)+"_"+str(columnCounter)][rowCounter] = angle

            columnCounter += 1
            if(columnCounter >= columnLimit):
                columnCounter = 0
                rowCounter += 1

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

    df.insert((columnLimit*handPointCount*2)+1,"Label", [targetLabel for i in range(rowCount)])
    print(df)
    df.to_csv('trainingData\\'+targetLabel+'_train.csv')

if __name__=="__main__":
    main()