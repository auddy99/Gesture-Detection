# Zexture (A Hand Gesture Detection Module)

![Project Image](https://i.imgur.com/QDD90Lk.jpeg)

---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [Working of project](#working-of-project)
- [References](#references)
- [Author Info](#author-info)

---

## Description

Jesture is openCV based hand gesture detection module. 

It uses the mediapipe library to detect hands by returning a set of 21 landmarks for each camera frame which contains a hand.

These 21 landmarks are extensively used by our team in order to build a module that can efficiently used by anyone to build IOT based applications which might need hand gestures to process instructions

### Technologies Used

- Python
- JSON

### Python Libraries Used

* OpenCV
* Mediapipe
* sklearn 
    - model_selection
    - metrics
    - ensemble
* Basic (Numpy, pandas, pickle)

---

## How To Use

### What to download?

You would require to clone this repository branch to your local device

Also you will need to run these commands in order to use the required libraries
- `pip install mediapipe`
- `pip install opencv-python`
- `pip install sklearn`
- `pip install pickle`

### What to do?

> Pictures are given in accordance to VSCode setup. You could follow in the same procedure for any other python runnable IDE

When you open the cloned repo, the folder structure would look something like this

![Project Image](https://i.imgur.com/zYiAC1v.jpeg)

You have been provided with a demo.py to see how you would be able to use the module methods

Run the file as it is to test the built-in gestures

These are all the built-in available gestures

![Project Image](https://i.imgur.com/OJiS5uV.png)

If it doesn't work, go to [Troubleshooting](#working-of-project) section.

All types of posible tweaks can be seen by hovering over `StaticGesture()` method or by going to the `modules/StaticGesture.py`

![Project Image](https://i.imgur.com/9s5F0dJ.jpeg)

### Lets make some training data 

Comment the `gesture.staticTest()` statement and uncomment the `gesture.addTrain("Your_Label")`

Replace the label string and run the file. It would run for 500 frames where your hand is visible.
> Make sure you move your hand gesture in a fasion so that it covers all perspectives of that hand gesture 

Now, reverse the last step by commenting the `gesture.addTrain("Your_Label")` statement and uncommenting the `gesture.staticTest()`

Run the `demo.py` file and you would be able to see the gesture recognised.

All the available gestures can seen in the `gestures.json` in `modules/assets`

### Let's move on to how can you use it for your own projects

A method called `testImage()` exists which takes in openCV image of `numpy.ndarray` type as parameter and returns a string label by using the model file `RFCModel.sav`

First use, delete the gestures you don't need from `modules/assets/staticTrainingData`, then use `addTrain()` method to input all gestures you want and check/test it using `staticTest()` 


---

## Working of Project

> ... Under Construction 🔨⚒🛠🚧🚧⚒🔨

## Troubleshooting

If the camera is not opening, you can try with different `cam` values of the `StaticGesture()`

Also you can use the `cameraTest()` method which is a simple check of camera compatibility


## References

This project is inspired from the code of [**Murtaza's Workshop**](https://www.youtube.com/watch?v=NZde8Xt78Iw&t). His [Youtube Channel](https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A) is extremely educational.

Big Thanks to his high quality code. Please take your time to check his videos if you want.

---

## Author Info

We are a group a three final year students pursuing B-Tech Degree.
- [Mayank Raj](https://www.linkedin.com/in/mayank-raj-2b51a3178/)
- [Nidhi Singh](https://www.linkedin.com/in/nidhisingh2010/)
- [Soumyadeep Auddy](https://www.linkedin.com/in/soumyadeep-auddy-270a89141/)
