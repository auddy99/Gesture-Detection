import cv2
import time
import sys
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import sqrt
def getCenterOfMass(lmList):
    sumX = 0
    for i in range(21):
        sumX = sumX + lmList[i][1]
    sumY = 0
    for i in range(21):
        sumY = sumY + lmList[i][2]

    return sumX/21, sumY/21
def pro_center_of_mass(val,frame):
    pos_x=0
    pos_y=0
    for i in range(frame):
        pos_x+=val[i][0]
        pos_y+=val[i][1]
    return pos_x/frame,pos_y/frame


def findDistance(x1,y1,x2,y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
def main():
    pTime = 0
    cTime = 0
    csv_val=pd.read_csv('trainingData.csv')
    frame=(len(csv_val.columns)-3)//21
    # print(frame)
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    dist_ant=[[(sys.maxsize,sys.maxsize,sys.maxsize) for i in range(21)] for i in range(frame)]
    value_in_model=[sys.maxsize]*(21*frame)
    loadedModel = pickle.load(open('firstModel.sav','rb'))
    pcheck=dict()
    pcheck[1]="come"
    pcheck[2]="go_awway"
    pcheck[3]="hiiiii"
    pcheck[4]="no"
    pcheck[5]="thumbs_up"
    pcheck[6]="turnaround"
    pcheck[7]="veritical_round"
    pcheck[8]="wave"
    answer=1
    flen=len
    fappend=value_in_model.append
    while True:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        if flen(lmlist) != 0:
            dist_ant.append(lmlist)
            dist_ant.pop(0)
            if dist_ant[0][0]!=(sys.maxsize,sys.maxsize,sys.maxsize):
                cx,cy=0,0
                t=0
                for i in dist_ant:
                    for j in i:
                        cx+=j[1]
                        cy+=j[2]
                        t+=1
                cx,cy=cx/t,cy/t
                for i in dist_ant:
                    for j in i:
                        fappend(findDistance(cx,cy,j[1],j[2]))
                        value_in_model.pop(0)
                if value_in_model[0]!=sys.maxsize:

                    answer = loadedModel.predict([value_in_model])
        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.putText(img,pcheck[np.int16(answer).item()],(100,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,103),3)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break
if __name__=="__main__":
    main()