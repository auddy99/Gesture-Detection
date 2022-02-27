import cv2
import time
import mediapipe as mp
from pkg_resources import get_distribution
import modules.HandTrackingModule as htm
import numpy as np
from math import *

def getDistance(pt1, pt2):
    return sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1]))

def getCrossProduct(pt1, pt2, pt3):
    p12 = getDistance(pt1, pt2)
    p13 = getDistance(pt1, pt3)
    p23 = getDistance(pt2, pt3)
    cosTheta = (p12*p12 + p13*p13 - p23*p23) / (2 * p12 * p13)
    sinTheta = sqrt(1 - cosTheta*cosTheta)
    res = p12*p13*sinTheta
    # print(str(p12) + " " + str(p13) + " " + str(sinTheta) + " " + str(res))
    return int(sinTheta*100)

def main():
    pTime = 0
    cTime = 0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    while True:
        success,img=cap.read()
        img=detector.findhands(img, draw=False)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            # for i in lmlist:
            #     print(str(i[0]) + ": (" + str(i[1]) + "," + str(i[2]) + ")\t")
            
            pt1 = (lmlist[0][1],lmlist[0][2])
            pt2 = (lmlist[4][1],lmlist[4][2])
            pt3 = (lmlist[20][1],lmlist[20][2])
            crossVectorRes = getCrossProduct(pt1, pt2, pt3)
            crossVector = (lmlist[0][1], crossVectorRes)

            cv2.arrowedLine(img, pt1, pt2, (0,0,0), 2)
            cv2.arrowedLine(img, pt1, pt3, (0,0,0), 2)
            cv2.arrowedLine(img, pt1, crossVector, (0,0,0), 2)
            

        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break
if __name__=="__main__":
    main()