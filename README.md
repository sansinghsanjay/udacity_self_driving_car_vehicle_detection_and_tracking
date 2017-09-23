# Udacity Self Driving Car Nanodegree: Vehicle Detection And Tracking

<p align="center">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_input_output/sample_input.gif">
&nbsp &nbsp
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/technologies_used/technologies_used.png">
&nbsp &nbsp
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_input_output/sample_output.gif">
</p>

## Objective
The objective of this project is to identify cars and track them in a video fram front facing camera of a Self Driving Car.

Above GIF images are showing sample input (at left side) and sample output (at right side) of this project, obtained by using different Machine Learning methods in Python.

## Introduction

### Self Driving Cars
Self Driving Cars are unmanned ground vehicles, also known as Autonomus Cars, Driverless Cars, Robotic Cars. 
<p align="center">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/images/self-driving-car.jpg">
</p>

### Technologies Used
Following are the technologies used by these Self Driving Cars to navigate:
1. Computer Vision and Machine Learning (AI) to find path, to classify traffic signs, etc.
2. Sensor Fusion to sense the surrounding moving vehicles.
3. Machine Learning (AI) for decision making.

### Why "Vehicle Detection And Tracking"?
Detecting and tracking other vehicles is important to prevent any accident. Every vehicle must keep a safe distance from each other in order to keep vehicle and passengers safe.

Here, I am using following three techniques to achieve aforementioned objective:
1. Random Forest
2. Support Vector Machine (SVM)
3. Convolutional Neural Network (CNN)

## Programming Language
In this project, Python-2.7.12 is used with following packages:
1. numpy - 1.13.0
2. moviepy - 0.2.3.2
3. cv2 - 3.0.0 (Computer Vision)
4. pandas - 0.19.0
5. sklearn - 0.18.1
6. tensorflow - 1.2.1
7. keras - 2.0.6

# Data Set Used
Data set is provided by Udacity. This data set is divided into two classes:
1. Vehicles: 8792 ".png" files (Dimension: 64x64)
2. Non-Vehicles: 8968 ".png" files (Dimension: 64x64)

Following are the sample images:
1. Vehicles [10-Images]:
<p align="center">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0007.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0036.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0067.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0194.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0214.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0305.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0750.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0843.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0876.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/vehicle/image0887.png">
</p>
2. Non-Vehicles: [10-Images]
<p align="center">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra1729.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra2.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra2476.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra26.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra423.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra5227.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/extra819.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/image216.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/image828.png">
<img src="https://github.com/sansinghsanjay/udacity_self_driving_car_vehicle_detection_and_tracking/blob/master/sample_data/non_vehicle/image93.png">
</p>

## Algorithm
1. Data Augmentation: Read every image and also generate its horizontal flip. Thus, it doubles the size of given data.
2. Generate Hog features, Spatial features and Histogram features of every image.
3. Normalize all the features generated in step 2 by using "sklearn.preprocessing.StandardScaler".
4. Apply 10-fold stratified cross validation on above generated features by using RandomForest and SVM model. Following are the results:
	[Accuracies Comparison: RF vs SVM]			[Kappa Comparison: RF vs SVM]
5. Finally, RandomForest and SVM are trained on entire train data.
6. Train a CNN (Architecture: two convolutional layer, one maxpooling layer, one dropout layer and one fully-connected layer, with "relu" as an activation function and "Adadelta" as optimizing function with learning rate 0.2 and trained for 50 epochs) on given data set images (without any augmentation). Following is the loss reduction plot of CNN:
			[cnn loss plot]
7. Finally all trained models (RandomForest, SVM, CNN), are tested on test images and test video. Following is the procedure of testing models on test data:
	1. Models were trained on 64x64 images but our test images and test videos are of size 1280x720. So, "Sliding Window Search" technique is used here with overlapping of (0.85, 0.85) to find out the location of vehicles. This window will move only on a particular area of image ("region of interest").
	2. After getting location of vehicles from previous step, to draw boxes over car, "multi-frame accumulated heatmap" (discussed below) technique is used here with previous 8 frames to reduce false positives in test video.

## Multi-frame Accumulated Heatmap
Multi-frame Accumulated Heatmap is a technique for creating a rectangular box over the object of interest in any given video. In this technique, we use a particular number of previous frames to get sure about the presence of object of interest. This also helps in removing false-positives.

Models are trained on 64x64 images but our test video are of 1280x720 size. So we work on a region of interest in each frame of test video.
Following is the input image (at left) and image with marked "area of interest" (at right):
[input image]	[marked_roi image]
Then, "Sliding Window Search" is applied on marked "region of interest" image. In this search, we extract 64x64 size image patches by moving a window over marked "region of interest" image. This window slides with overlap of 0.85, horizontally and vertically.
Image patches which are classified as "vehicle" by our trained model, are used for creating a heatmap:
[heatmap image]
Then, a threshold is applied on these heatmaps to remove false-positives:
[threshold applied image]
Then, I applied "label" operation of "scipy.ndimage.measurements". Its a Computer Vision operation to detect connected regions in a binary digital image.
[label image]
With the help of above operation, we got pixel values of all connected regions. With the help of this, now we can draw box over object of interest:
[output image]

## How To Use?
To use this project:
1. Clone this github repository.
2. Make "scripts" sub-directory as present-working-directory.
3. Run this command in terminal: ```python main.py```
Or simply use ipython notebook under "scripts" sub-directory.

## Limitations:
1. It was very hard to make model distinguishing between black car and shadow.
2. Model seems to be giving lots of false positives for objects in dim light like shadow, under tunnels.
