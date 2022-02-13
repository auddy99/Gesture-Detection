# import the Static Gesture Module 
import modules.StaticGesture as sg

def main():
    # - Create an object that will be used to perform all the functionalities
    gesture = sg.StaticGesture()
    # Parameters
    # ----------
    # `cam` : int (default = 0)
    #     Which camera device will be used
    # `dataloc` : string path (default = r'modules\staticTrainingData')
    #     location where all training data and other assets are stored
    # `trainName` : string (default = 'staticData')
    #     FileName of final training data
    # `modelName` : string (deafult = 'RFCModel')
    #     Exported model name

    # - Test whether openCV camera is working properly
    # gesture.cameraTest()

    # - Use the following method to train data with a target Label
    # gesture.addTrain("Your_Label")

    # - Test the trained model in real-time camera
    gesture.staticTest()

    # PRESS ESCAPE KEY IN ORDER TO CLOSE THE CAMERA

if __name__=="__main__":
    main()