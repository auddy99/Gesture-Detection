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

    def __init__(self,  cam=0, dataLoc=r"modules\assets", trainLoc=r"modules\assets\staticTrainingData", trainName="staticData", modelName="RFCModel"):
        """
        Object to set up static gesture operations

        Parameters
        ----------
        `cam` : int (default = 0)
            Which camera device will be used
        `dataloc` : string path (default = r'modules\assets')
            location where all asset related data and other assets are stored
        `trainloc` : string path (default = r'modules\assets\staticTrainingData')
            location where all training data and other assets are stored
        `trainName` : string (default = 'staticData')
            FileName of final training data
        `modelName` : string (deafult = 'RFCModel')
            Exported model name

        See Also
        --------
        `StaticGesture.cameraTest` : Test whether openCV can open your camera properly
        `StaticGesture.staticTrain` : Train data with your own gestures
        `StaticGesture.joinTrainingSets` : Combine all training data to one file
        `StaticGesture.modelRFC` : Apply Random Forest Regression to create model 
        `StaticGesture.addTrain` : Combine `staticTrain()`, `joinTrainingSets()`, `modelRFC()` into single method
        `StaticGesture.testImage` : Test a single image frame and return result
        `StaticGesture.staticTest` : Open Camera and Test the model real-time
        """
        self.dataLoc = dataLoc
        self.trainLoc = trainLoc
        self.detector = htm.handDetector()
        self.cam = cam
        self.trainName = trainName
        self.modelName = modelName

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

    def staticTrain(self, targetLabel, sampleSize=500):
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

        p = dict()
        p['Label'] = [targetLabel for i in range (sampleSize)]
        p['size_ratio'] = []
        for i in range(21):
            p[str(i)+'_dist'] = []
            p[str(i)+'_angle'] = []

        while countLabel < sampleSize:
            success,img = cap.read()
            img = self.detector.findhands(img, draw=False)
            lmlist = self.detector.findPosition(img)
            
            if len(lmlist) != 0:

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

                p['size_ratio'].append(boxLength / boxHeight)

                for i in range(21):
                    distFromCenter, angleFromCenter = getVector(center, (lmlist[i][1], lmlist[i][2]))
                    if str(i)+'_dist' in p:
                        p[str(i)+'_dist'].append(distFromCenter/boxDiagonal)
                        p[str(i)+'_angle'].append(angleFromCenter)

                countLabel = countLabel+1

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
            cv2.putText(img, "Frames taken: "+str(countLabel), (310,30), cv2.FONT_HERSHEY_PLAIN, 2, (150,0,0), 2)
            cv2.imshow('image1',img)
            
            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break


        df = pd.DataFrame(p)
        saveLoc = self.trainLoc+'\\'+targetLabel+'_data.csv'
        df.to_csv(saveLoc)

    def joinTrainingSets(self, selectedHandPoints = [0,4,8,12,16,20]):
        """
        Combine all training data to one file

        Takes all training data files from `self.dataloc` location
        """
        all_files = glob.glob(self.trainLoc + "/*_data.csv")
        jsonData = {}
        with open(self.dataLoc + "\\gestures.json", 'r') as f:
            jsonData = json.load(f)
            jsonData["gestures"] = []
        
        req_cols = ['Label', 'size_ratio']
        for i in selectedHandPoints:
            req_cols.append(str(i) + '_dist')
            req_cols.append(str(i) + '_angle')

        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df[req_cols]
            jsonData["gestures"].append(df["Label"][0])
            li.append(df)
        
        self.gestures = jsonData['gestures']
        with open(self.dataLoc + "\\gestures.json", 'w') as f:
            jsonData["selectedHandPoints"] = selectedHandPoints
            json.dump(jsonData, f)

        self.gestureCount = len(li)
        frame = pd.concat(li, axis=0, ignore_index=True)
        saveLoc = self.dataLoc+'\\'+self.trainName+".csv"
        frame.to_csv(saveLoc)

    def modelRFC(self):
        """
        Apply Random Forest Regression to create model 

        Exports model by the name of `self.modelName`
        """
        df = pd.read_csv(self.dataLoc+'\\'+"staticData"+".csv")
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
        print(y.value_counts())
        rfc.fit(X.values,y.values)
        self.model = rfc
        pickle.dump(rfc, open(self.dataLoc + "\\" + self.modelName +'.sav', 'wb'))

    def addTrain(self, targetLabel, sampleSize = 500):
        """
        Combine `staticTrain()`, `joinTrainingSets()`, `modelRFC()` into single method

        Parameters
        ----------
        `targetLabel` : string
            Name of the gesture you want to return
        `sampleSize` : int
            - Number of rows of training data. 
            - More sampleSize means more accuracy but it takes more time to train
            - `Warning`: Using different sampleSize for different training data might cause mismatch and lead to unexpected results
        """
        self.staticTrain(targetLabel, sampleSize)
        self.joinTrainingSets()
        self.modelRFC()

    def testImage(self, img, show, selectedHandPoints):
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

        if len(lmlist) != 0:

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

            testList = [boxLength / boxHeight]
            for i in range(21):
                if(i not in selectedHandPoints):
                    continue
                distFromCenter, angleFromCenter = getVector(center, (lmlist[i][1], lmlist[i][2]))
                testList.append(distFromCenter/boxDiagonal)
                testList.append(angleFromCenter)
            
            answer = self.model.predict([testList])
            result = self.gestures[int(answer)]
            return result
        else:
            return ""
    
    def staticTest(self, show=True):
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
        
        while True:
            success,img = cap.read()
            answer = self.testImage(img, show, jsonData['selectedHandPoints'])

            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime=cTime
            
            cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
            cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
            cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
            cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)

            if answer != "":
                cv2.putText(img, answer, (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            else:
                cv2.putText(img, " (No Hands Detected)", (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

            cv2.imshow('image1',img)

            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break


# remove the modules parent from initial imports to use the below main method 

# def main():
#     sg = StaticGesture("Flat", 500)
#     # sg.cameraTest()
#     # sg.staticTrain()
#     # sg.joinTrainingSets()
#     # sg.modelRFC()
#     sg.staticTest()

# if __name__=="__main__":
#     main()