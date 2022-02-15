import cv2
import time
import mediapipe as mp
# import matplotlib.pyplot as plt
class handDetector():
    def __init__(self, mode=False, maxhands=1, detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxhands,1,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)

                # for id,ln in enumerate(handlms.landmark):
                    # print(id,ln)
                    # h,w,c = img.shape
                    # cx,cy=int(ln.x*w),int(ln.y*h)
                    # print(id,cx,cy)
                    # if id==15:
                    # cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return img
        
    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 7, (255,0,255), cv2.FILLED)
                # if id in [1,2,3,4]:
                #     cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
                # if id in [5,6,7,8]:
                #     cv2.circle(img, (cx, cy), 15, (0,255,255), cv2.FILLED)

                # cv2.putText(img,"hello",(30,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        return lmList


# def main():
#     pTime=0
#     cTime=0
#     cap=cv2.VideoCapture(0)
#     detector=handDetector()
#     while True:
#         success,img=cap.read()
#         img=detector.findhands(img)
#         lmlist = detector.findPosition(img)
#         if len(lmlist) != 0:
#             print(lmlist[4])

#         cTime=time.time()
#         fps=1/(cTime-pTime)
#         pTime=cTime

#         cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
#         cv2.imshow('image1',img)
#         keyPressed = cv2.waitKey(5)
#         # if keyPressed == ord('q'):
#         #     break;
# if __name__=="__main__":
#     main()