# libraries
from math import *
import cv2
import os
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Cropping2D
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

### model prediction ###
# 0: CAR
# 1: NON-CAR
########################

# setting random seed
np.random.seed(77)

# global variables
orientation = 8
pixels_per_cell = 8
cells_per_block = 2
spatial_size = (16, 16)
histogram_nbins = 32
rectangle_pt1 = (525, 370)
rectangle_pt2 = (1280, 660)
window_size = (64, 64)
window_overlapping = (0.85, 0.85)
train_image_size = (64, 64)
heatmap_threshold = 5
heat_prev = np.zeros((720, 1280)) # to store previous heatmap
prev_heat_count = 0

# paths
test_images_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/test_images/"
test_images_output_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/test_images_output/"
vehicle_images_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/data/vehicles/vehicles/"
non_vehicle_images_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/data/non-vehicles/non-vehicles/"
scaler_object_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/trained_model/scaler_object.pkl"
trained_rf_model_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/trained_model/rf_model.pkl"
trained_svm_model_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/trained_model/svm_model.pkl"
cnn_model_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/trained_model/cnn/cnn.ckpt"
project_video_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/project_video/project_video.mp4"
project_video_output_path = "/home/local/ALGOANALYTICS/sanjay/keep/vehicle_detection/project_video_output/project_video_output.mp4"

# function for reading image
def read_image(image_path):
	# reading image and converting from BGR to RGB
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

# function for showing image
def show_image(img, title=None):
	plt.imshow(img)
	if(title != None):
		plt.title(title)
	plt.show()

# function for drawing rectangle on image
def draw_rectangle(img, top_left_corner, bottom_right_corner):
	# drawing rectangle
	cv2.rectangle(img, top_left_corner, bottom_right_corner, (0,0,255), 2)
	return img

# function for preparing train data
def get_data(path):
	# for collecting path of all images
	all_images_path = []
	# getting list of subdirs
	subdir_list = os.listdir(path)
	# traversing through each subdirectory
	for i in range(len(subdir_list)):
		# getting list of files
		files = os.listdir(path + subdir_list[i] + "/")
		# traversing through each file
		for j in range(len(files)):
			# path of current file
			file_path = path + subdir_list[i] + "/" + files[j]
			# appending file_path
			all_images_path.append(file_path)
	return all_images_path

# function for getting hog features
def get_hog_features(img):
	global orientation, pixels_per_cell, cells_per_block
	# getting hog-features
	features = hog(img, orientations=orientation,
				pixels_per_cell=(pixels_per_cell, pixels_per_cell),
				cells_per_block=(cells_per_block, cells_per_block), 
				transform_sqrt=True, visualise=False, feature_vector=True)
	return features

# function for getting spatial features
def get_spatial_features(img):
	global spatial_size
	return cv2.resize(img, spatial_size).ravel()

# function for getting color histogram features
def get_histogram_features(img):
	global histogram_nbins
	channel_1 = np.histogram(img[:,:,0], bins=histogram_nbins, range=(0, 256))[0]
	channel_2 = np.histogram(img[:,:,1], bins=histogram_nbins, range=(0, 256))[0]
	channel_3 = np.histogram(img[:,:,2], bins=histogram_nbins, range=(0, 256))[0]
	histogram = np.hstack((channel_1, channel_2, channel_3))
	return histogram

# function for reading all images, generating and returning all images
def get_all_features_labels(path_list):
	# variable for storing features
	feature_set = []
	labels = []
	# traversing through all paths in path1
	for i in range(len(path_list)):
		# reading image
		img = read_image(path_list[i])
		# flipping image (data augmentation)
		flipped_img = cv2.flip(img, 1)
		# generating features for original image
		img_hog_features = get_hog_features(cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:,:,0])
		img_spatial_features = get_spatial_features(img)
		img_hist_features = get_histogram_features(img)
		img_features = np.hstack((img_hog_features, img_spatial_features, img_hist_features))
		# generating features for flipped image
		flipped_img_hog_features = get_hog_features(flipped_img[:,:,0])
		flipped_img_spatial_features = get_spatial_features(flipped_img)
		flipped_img_hist_features = get_histogram_features(flipped_img)
		flipped_img_features = np.hstack((flipped_img_hog_features, flipped_img_spatial_features, flipped_img_hist_features))
		# appending features into main list
		feature_set.append(img_features)
		feature_set.append(flipped_img_features)
		# making label
		if(path_list[i].split("/")[8] == "vehicles"):
			labels.append(0)
			labels.append(0)
		elif(path_list[i].split("/")[8] == "non-vehicles"):
			labels.append(1)
			labels.append(1)
		else:
			print("Error in get_all_features_labels")
			break
	return np.array(feature_set), np.array(labels)

def sliding_window_positions():
	global rectangle_pt1, rectangle_pt2, window_size, window_overlapping
	# calculating length of region of interest
	len_x_roi = rectangle_pt2[0] - rectangle_pt1[0]
	len_y_roi = rectangle_pt2[1] - rectangle_pt1[1]
	# calculating number of pixels by which window slides
	x_inc = np.int(window_size[0] * (1 - window_overlapping[0]))
	y_inc = np.int(window_size[1] * (1 - window_overlapping[1]))
	# calculating number of windows along x-axis
	x_overlap = np.int(window_size[0] * window_overlapping[0])
	no_x_windows = np.int((len_x_roi - x_overlap) / x_inc)
	# calculating number of windows along y-axis
	y_overlap = np.int(window_size[1] * window_overlapping[1])
	no_y_windows = np.int((len_y_roi - y_overlap) / y_inc)
	# list variable for storing position of windows
	windows = []
	# loop through finding x and y window positions
	for y_window in range(no_y_windows):
		for x_window in range(no_x_windows):
			# calculating window position
			x_start = x_window * x_inc + rectangle_pt1[0]
			x_end = x_start + window_size[0]
			y_start = y_window * y_inc + rectangle_pt1[1]
			y_end = y_start + window_size[1]
			# appdinding positions in list
			windows.append(((x_start, y_start), (x_end, y_end)))
	# return the list of windows
	return windows

# function for drawing windows on images
def draw_windows(img, windows):
	# iterating through windows
	for i in range(len(windows)):
		# drawing rectangle on image
		img = draw_rectangle(img, windows[i][0], windows[i][1])
	return img

# function for finding windows probably having car
def find_windows_having_car(img, windows, scaler, model):
	global train_image_size
	# for listing windows probably having car
	car_windows = []
	# iterating windows
	for i in range(len(windows)):
		# extracting window
		image_patch = img[windows[i][0][1]:windows[i][1][1], windows[i][0][0]:windows[i][1][0], :]
		# resizing image_patch to train_image_size
		image_patch = cv2.resize(image_patch, train_image_size)
		# if scaler is not-none, using randomForest or SVM
		if(scaler != None):
			# genrating features
			hog_features = get_hog_features(image_patch[:,:,0])
			spatial_features = get_spatial_features(image_patch)
			hist_features = get_histogram_features(image_patch)
			features = np.hstack((hog_features, spatial_features, hist_features))
			# normalizing features
			features = scaler.transform(features)
			# getting prediction: car or non-car
			pred = model.predict(features)
			# if car is present 
			if(pred == 0):
				car_windows.append(windows[i])
		# else using SVM
		else:
			# reshaping image matrix
			image_patch = np.reshape(image_patch, (1, train_image_size[0], train_image_size[1], 3))
			pred = model.predict(image_patch, batch_size=1)
			pred = np.argmax(pred, 1)
			if(pred == 0):
				car_windows.append(windows[i])
	return car_windows

# function for adding heat in heatmap image
def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	# Return updated heatmap
	return heatmap

# function for applying threshold on heat image
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

# function for marking cars
def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img

# function for loading pretrained CNN model
def load_cnn():
	input_shape = (64, 64, 3)
	learning_rate = 0.2
	# cnn model
	model = Sequential()
	model.add(Lambda(lambda x:x/255.0, input_shape=input_shape))
	model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
	model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(2))
	# CNN optimizer
	optimizer = optimizers.Adadelta(lr=learning_rate)
	# compiling model
	model.compile(loss="binary_crossentropy", optimizer=optimizer)
	# load model
	model.load_weights(cnn_model_path)
	return model

# function for processing video
def pipeline(img):
	global windows, cnn_model, heatmap_threshold, heat_prev, prev_heat_count
	# finding windows probably having car
	car_windows = find_windows_having_car(img, windows, None, cnn_model)
	# heatmap image
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# add heat to each window in windows list
	heat = add_heat(heat, car_windows)
	# multiframe accumulated heatmap
	heat_out = heat_prev + heat
	prev_heat_count += 1
	if(prev_heat_count >= 8):
		heat_prev = heat
		prev_heat_count = 0
	# apply threshold to help remove false positives
	heat = apply_threshold(heat_out, heatmap_threshold)
	# clipping all values in heatmap
	heatmap = np.clip(heat, 0, 255)
	# find final boxes from heatmap using label function
	labels = label(heatmap)
	# drawing car_windows on image and showing image
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	return draw_img

'''# getting path of all vehicle and non-vehicle images
vehicle_images_path_list = get_data(vehicle_images_path)
non_vehicle_images_path_list = get_data(non_vehicle_images_path)
images_list = np.array(vehicle_images_path_list + non_vehicle_images_path_list)

# getting features and labels
features, labels = get_all_features_labels(images_list)

# preprocessing features
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

# saving scaler object
joblib.dump(scaler, scaler_object_path)

# updating status
print("Features Shape: " + str(features.shape))
print("Labels Shape: " + str(len(labels)))

# object for 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, random_state=77)

# 10-fold cross-validation
fold = 1
for train_index, val_index in skf.split(features, labels):
	# dividing data into train and validation set
	train_X, val_X = features[train_index], features[val_index]
	train_Y, val_Y = labels[train_index], labels[val_index]
	# getting models
	rf_model = RandomForestClassifier()
	svm_model = LinearSVC(loss='hinge')
	# training model
	rf_model.fit(train_X, train_Y)
	svm_model.fit(train_X, train_Y)
	# validating model
	rf_pred = rf_model.predict(val_X)
	svm_pred = svm_model.predict(val_X)
	# getting accuracy and kappa
	rf_accuracy = accuracy_score(rf_pred, val_Y)
	svm_accuracy = accuracy_score(svm_pred, val_Y)
	rf_kappa = cohen_kappa_score(rf_pred, val_Y)
	svm_kappa = cohen_kappa_score(svm_pred, val_Y)
	# updating status
	print(str(fold) + ". RF Accuracy: " + str(rf_accuracy) + "\tRF Kappa: " + str(rf_kappa))
	print(str(fold) + ". SVM Accuracy: " + str(svm_accuracy) + "\tSVM Kappa: " + str(svm_kappa))
	fold = fold + 1

# getting models
rf_model = RandomForestClassifier()
svm_model = LinearSVC(loss='hinge')

# training models on entire data
rf_model.fit(features, labels)
svm_model.fit(features, labels)

# saving model
joblib.dump(rf_model, trained_rf_model_path)
joblib.dump(svm_model, trained_svm_model_path)'''

# loading scaler object, random forest model and svm model
#scaler = joblib.load(scaler_object_path)
#rf_model = joblib.load(trained_rf_model_path)
#svm_model = joblib.load(trained_svm_model_path)
cnn_model = load_cnn()

# getting list of windows (sliding window)
windows = sliding_window_positions()

# processing all test images
test_files = os.listdir(test_images_path)
for i in range(len(test_files)):
	start_time = time.time()
	# reading test image
	img = read_image(test_images_path + test_files[i])
	# finding windows probably having car
	#rf_car_windows = find_windows_having_car(img, windows, scaler, rf_model)
	#svm_car_windows = find_windows_having_car(img, windows, scaler, svm_model)
	cnn_car_windows = find_windows_having_car(img, windows, None, cnn_model)
	# drawing car_windows on image
	#rf_car_windows_img = draw_windows(np.copy(img), rf_car_windows)
	#svm_car_windows_img = draw_windows(np.copy(img), svm_car_windows)
	cnn_car_windows_img = draw_windows(np.copy(img), cnn_car_windows)
	# heatmap image
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# add heat to each window in windows list
	#heat = add_heat(heat, svm_car_windows)
	heat = add_heat(heat, cnn_car_windows)
	# apply threshold to help remove false positives
	heat = apply_threshold(heat, heatmap_threshold)
	# clipping the values in heat image
	heatmap = np.clip(heat, 0, 255)
	# find final boxes from heatmap using label function
	labels = label(heatmap)
	# draw boxes over car
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	time_taken = time.time() - start_time
	print("Time Taken: " + str(time_taken))
	# showing images
	#show_image(rf_car_windows_img, title="RF")
	#show_image(svm_car_windows_img, title="SVM")
	#show_image(cnn_car_windows_img, title="CNN")
	#show_image(draw_img, title="Heatmap Image")
	# saving output image
	draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
	cv2.imwrite(test_images_output_path + test_files[i], draw_img)

# processing project video
video_input1 = VideoFileClip(project_video_path)
processed_video = video_input1.fl_image(pipeline)
processed_video.write_videofile(project_video_output_path, audio=False)
