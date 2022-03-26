from matplotlib.pyplot import axis
import modules.HandTrackingModule as htm
from modules.GestureMath import *

import cv2
import time
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
import glob
from math import *
import json 
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class Zexture:

    def __init__(self,  cam=0, dataLoc=r"modules\assets", trainLoc=r"modules\assets\dynamicTrainingData", trainName="dynamicData", modelName="RFCModel"):
        """
        Object to set up dynamic gesture operations

        Parameters
        ----------
        `cam` : int (default = 0)
            Which camera device will be used
        `dataloc` : string path (default = r'modules\assets')
            location where all asset related data and other assets are stored
        `trainloc` : string path (default = r'modules\assets\dynamicTrainingData')
            location where all training data and other assets are stored
        `trainName` : string (default = 'dynamicData')
            FileName of final training data
        `modelName` : string (deafult = 'RFCModel')
            Exported model name

        See Also
        --------
        `DynamicGesture.cameraTest` : Test whether openCV can open your camera properly
        `DynamicGesture.dynamicTrain` : Train data with your own gestures
        `DynamicGesture.joinTrainingSets` : Combine all training data to one file
        `DynamicGesture.modelRFC` : Apply Random Forest Regression to create model 
        `DynamicGesture.addTrain` : Combine `dynamicTrain()`, `joinTrainingSets()`, `modelRFC()` into single method
        `DynamicGesture.testImage` : Test a single image frame and return result
        `DynamicGesture.dynamicTest` : Open Camera and Test the model real-time
        """
        self.dataLoc = dataLoc
        self.trainLoc = trainLoc
        self.detector = htm.handDetector()
        self.cam = cam
        self.trainName = trainName
        self.modelName = modelName

        # Testing Variables
        self.counter = 0
        self.frameLimit = 5
        self.testList = []
        self.answer = -1

        try:
            with open(self.dataLoc + "\\gestures.json", 'r') as f:
                data = json.load(f)
            self.gestures = data['gestures']
        except:
            print("Gesture File Not Found")

        try:
            self.model = pickle.load(open(self.dataLoc + "\\" + modelName +'.sav','rb'))
        except:
            print("Model File Not Found")

    def cameraTest(self, showHand=False):
        """
        Test whether openCV can open your camera properly
        Camera device number can be changed during Object Initialisation

        Parameters
        ----------
        `showHand` : boolean
            Shows what skeleton the camera is picking up 
        """
        cap = cv2.VideoCapture(self.cam)
        while True:
            success,img = cap.read()
            print(type(img))
            if showHand==True:
                img = self.detector.findhands(img)
            
            cv2.putText(img, str(random.randint(1,10)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            cv2.imshow('image1',img)

            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break

    def dynamicTrain(self, targetLabel, sampleSize=500):
        """
        Train data with your own gestures

        Parameters
        ----------
        `targetLabel` : string
            Name of the gesture you want to return
        `sampleSize` : int
            - Number of rows of training data. 
            - More sampleSize means more accuracy but it takes more time to train
            - `Warning`: Using different sampleSize for different training data might cause mismatch and lead to unexpected results

        """
        pTime,cTime = 0,0
        cap = cv2.VideoCapture(self.cam)
        countLabel = 0

        column_names = []
        column_names.append("Size_Ratio")
        for j in range(21):
            column_names.append(str(j)+"_dist")
            column_names.append(str(j)+"_angle")
            column_names.append(str(j)+"_distC")
            column_names.append(str(j)+"_angleC")
        data = [[0.0 for j in range(85)] for i in range(sampleSize)]
        df = pd.DataFrame(data, columns=column_names)
        prevlmlist = [[0,0,0] for i in range(21)]

        while countLabel < sampleSize:

            success,img = cap.read()
            img = self.detector.findhands(img)
            lmlist = self.detector.findPosition(img)

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

                df["Size_Ratio"][countLabel] = boxLength / boxHeight
                for i in range(21):
                    cv2.arrowedLine(img, center, (lmlist[i][1], lmlist[i][2]), (0,0,0), 2)
                    cv2.arrowedLine(img, (prevlmlist[i][1], prevlmlist[i][2]), (lmlist[i][1], lmlist[i][2]), (255,255,255), 2)
                    dist, angle = getVector(center,(lmlist[i][1], lmlist[i][2]))
                    distC, angleC = getVector((prevlmlist[i][1], prevlmlist[i][2]),(lmlist[i][1], lmlist[i][2]))
                    df[str(i)+"_dist"][countLabel] = dist/boxDiagonal
                    df[str(i)+"_angle"][countLabel] = angle
                    df[str(i)+"_distC"][countLabel] = distC/boxDiagonal
                    df[str(i)+"_angleC"][countLabel] = angleC
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

        df.insert(0,"Label", [targetLabel for i in range(sampleSize)])
        # print(df)
        saveLoc = self.trainLoc+'\\'+targetLabel+'_data.csv'
        df.to_csv(saveLoc)

    def joinTrainingSets(self, selectedHandPoints = [0,4,8,12,16,20], frameLimit = 5):
        """
        Combine all training data to one file

        Takes all training data files from `self.dataloc` location
        """

        all_files = glob.glob(self.trainLoc + "/*_data.csv")
        self.frameLimit = frameLimit
        # JSON update
        
        jsonData = {}
        final_df = []
        with open(self.dataLoc + "\\gestures.json", 'r') as f:
            jsonData = json.load(f)
            jsonData["gestures"] = []

        filter_cols = ['Label','Size_Ratio']
        for i in selectedHandPoints:
            filter_cols.append(str(i)+"_dist")
            filter_cols.append(str(i)+"_angle")
            filter_cols.append(str(i)+"_distC")
            filter_cols.append(str(i)+"_angleC")
        
        req_cols = ['Label']
        for i in range(self.frameLimit):
            req_cols.append(str(i)+"_Size_Ratio")
            for j in selectedHandPoints:
                req_cols.append(str(j)+"_"+str(i)+'_dist')
                req_cols.append(str(j)+"_"+str(i)+'_angle')
                req_cols.append(str(j)+"_"+str(i)+'_distC')
                req_cols.append(str(j)+"_"+str(i)+'_angleC')

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df[filter_cols]
            label = df['Label'][0]
            jsonData["gestures"].append(label)
            df = df.drop('Label', axis=1)

            full_row = [label]
            for index, row in df.iterrows():
                full_row = full_row + list(row)
                if index % frameLimit == frameLimit-1:
                    final_df.append(full_row)
                    full_row = [label]
        
        self.gestures = jsonData['gestures']
        with open(self.dataLoc + "\\gestures.json", 'w') as f:
            jsonData["selectedHandPoints"] = selectedHandPoints
            jsonData["frameLimit"] = frameLimit
            json.dump(jsonData, f)
        self.gestureCount = len(final_df)

        frame = pd.DataFrame(final_df, columns=req_cols)
        print(frame)
        saveLoc = self.dataLoc+'\\'+self.trainName+".csv"
        frame.to_csv(saveLoc)

    def modelRFC(self):
        """
        Apply Random Forest Regression to create model 

        Exports model by the name of `self.modelName`
        """
        df = pd.read_csv(self.dataLoc+'\\'+"dynamicData"+".csv")
        df = df.iloc[: , 1:]
        print(df)

        label_encoder = preprocessing.LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])

        y = df['Label'].astype('int')
        X = df.drop('Label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

        rfc = RandomForestClassifier(n_estimators=600)
        rfc.fit(X_train,y_train)

        predictions = rfc.predict(X_test)
        print(classification_report(y_test,predictions))
        rfc.fit(X.values, y.values)
        self.model = rfc
        pickle.dump(rfc, open(self.dataLoc + "\\" + self.modelName +'.sav', 'wb'))

    def addTrain(self, targetLabel, sampleSize = 500, frameLimit = 5):
        """
        Combine `dynamicTrain()`, `joinTrainingSets()`, `modelRFC()` into single method

        Parameters
        ----------
        `targetLabel` : string
            Name of the gesture you want to return
        `sampleSize` : int
            - Number of rows of training data. 
            - More sampleSize means more accuracy but it takes more time to train
            - `Warning`: Using different sampleSize for different training data might cause mismatch and lead to unexpected results
        """
        self.dynamicTrain(targetLabel, sampleSize)
        self.joinTrainingSets(frameLimit=frameLimit)
        self.modelRFC()

    def testImage(self, img, show, selectedHandPoints, resultList, prevlmlist):
        """
        Test a single image frame and return result

        Parameters
        ----------
        `img` : openCV cap.read() returned image of `numpy.ndarray` type
            Image which be tested
        `show` : boolean (default = True)
            Shows the hand skeleton while viewing
        """
        
        img = self.detector.findhands(img, draw=True)
        lmlist = self.detector.findPosition(img)

        if self.counter % self.frameLimit == 0 and len(self.testList) != 0:
            self.answer = self.model.predict([self.testList])
            self.testList = []
            self.counter = 0

        if len(lmlist):

            x_list = [i[1] for i in lmlist]
            y_list = [i[2] for i in lmlist]

            origin = (min(x_list), min(y_list))
            terminal = (max(x_list), max(y_list))
            boxLength = terminal[0] - origin[0]
            boxHeight = terminal[1] - origin[1]
            boxDiagonal = sqrt(boxLength*boxLength + boxHeight*boxHeight)
            center = ((int)(origin[0]+boxLength/2), (int)(origin[1]+boxHeight/2))

            # cv2.rectangle(img, origin, terminal, color=(0,0,255), thickness=2)
            # cv2.circle(img, origin, 3, (255,0,0), cv2.FILLED)
            # cv2.circle(img, terminal, 3, (255,0,0), cv2.FILLED)
            # cv2.circle(img, center, 5, (0,255,0), cv2.FILLED)

            self.testList.append(boxLength / boxHeight)
            for i in selectedHandPoints:
                targetPoint = (lmlist[i][1], lmlist[i][2])
                prevPoint = (prevlmlist[i][1], prevlmlist[i][2])

                # cv2.arrowedLine(img, center, targetPoint, (0,0,0), 2)
                # colorTrail = 0
                # for j in trail[i]:
                #     if showTrail:
                #         cv2.line(img, j[0], j[1], (colorTrail,255-colorTrail,colorTrail), 2)
                #     colorTrail += 30
                # # cv2.arrowedLine(img, prevPoint, targetPoint, (0,255,0), 2)
                # trail[i].append((prevPoint, targetPoint))
                # if(len(trail[i]) > 10):
                #     trail[i].pop(0)
                

                dist, angle = getVector(center,targetPoint)
                distC, angleC = getVector(prevPoint,targetPoint)
                self.testList.append(dist/boxDiagonal)
                self.testList.append(angle)
                self.testList.append(distC/boxDiagonal)
                self.testList.append(angleC)
            self.counter += 1
            prevlmlist = lmlist
            if self.answer == -1:
                return "None"
            return resultList[int(self.answer)]

        else:
            return " (No Hands Detected)"

    
    def dynamicTest(self, show=True):
        """
        Start camera and test the model real-time

        Parameters
        ----------
        `show` : boolean (default = True)
            Shows the hand skeleton while viewing
        """
        pTime,cTime = 0,0
        cap = cv2.VideoCapture(self.cam)        

        with open(self.dataLoc + "\\gestures.json", 'r') as f:
            jsonData = json.load(f)

        prevlmlist = [[0,0,0] for i in range(21)]
        
        while True:
            success,img = cap.read()
            result = self.testImage(img, show, jsonData['selectedHandPoints'], jsonData['gestures'], prevlmlist)
            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime=cTime
            
            cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
            cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
            cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
            cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)

            if result != "":
                cv2.putText(img, result, (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            else:
                cv2.putText(img, " (No Hands Detected)", (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

            cv2.imshow('image1',img)

            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                self.counter = 0
                self.testList = []
                self.answer = -1
                break


# remove the modules parent from initial imports to use the below main method 

# def main():
#     sg = DynamicGesture("Flat", 500)
#     # sg.cameraTest()
#     # sg.dynamicTrain()
#     # sg.joinTrainingSets()
#     # sg.modelRFC()
#     sg.dynamicTest()

# if __name__=="__main__":
#     main()