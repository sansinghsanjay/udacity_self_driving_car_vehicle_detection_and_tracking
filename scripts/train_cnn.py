# libraries
import os
import cv2
import math
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Cropping2D
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda

### model prediction ###
# 0: CAR
# 1: NON-CAR
########################

# setting random seed
np.random.seed(77)

# global variables and constants
input_shape = (64, 64, 3)
batch_size = 100.0
learning_rate = 0.2
epochs = 50
len_train_X = 0
len_val_X = 0

# paths
test_images_path = "/home/sansingh/sanDocs/certs/udacity_selfDrivingCar/part1/VehicleDetectionAndTracking/submission/test_images/"
vehicle_images_path = "/home/sansingh/sanDocs/certs/udacity_selfDrivingCar/part1/VehicleDetectionAndTracking/submission/data/vehicles/vehicles/"
non_vehicle_images_path = "/home/sansingh/sanDocs/certs/udacity_selfDrivingCar/part1/VehicleDetectionAndTracking/submission/data/non-vehicles/non-vehicles/"
cnn_model_path = "/home/sansingh/sanDocs/certs/udacity_selfDrivingCar/part1/VehicleDetectionAndTracking/submission/trained_model/cnn/cnn.ckpt"

# function for reading image
def read_image(image_path):
	# reading image
	img = cv2.imread(image_path)
	# transforming image color model
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

# function for reading image pixels and making labels
def read_images_make_labels(imges_path_list):
	# variable for storing all images and labels
	images_list = []
	labels_list = []
	# iterating through each image
	for i in range(len(images_path_list)):
		# reading image
		img = read_image(images_path_list[i])
		# appending image
		images_list.append(img)
		# making label
		if(images_path_list[i].split("/")[11] == "vehicles"):
			labels_list.append(0)
		elif(images_path_list[i].split("/")[11] == "non-vehicles"):
			labels_list.append(1)
		else:
			print("Error: labels not found")
	return np.array(images_list), np.array(labels_list)

# generator function for training
def train_generator():
	try:
		# global variables
		global batch_size, input_shape, train_X, train_Y, len_train_X
		# getting length of train data
		len_train_X = len(train_X)
		# converting labels into one-hot vector
		train_Y = np.array(pd.get_dummies(pd.Series(train_Y)))
		# runing generator
		while(True):
			for index in range(int(math.ceil(len_train_X/batch_size))):
				# making batches of size batch_size
				current_image_bucket = train_X[index * int(batch_size) : (index + 1) * int(batch_size), :, :, :]
				current_label_bucket = train_Y[index * int(batch_size) : (index + 1) * int(batch_size), :]
				yield current_image_bucket, current_label_bucket
	except Exception as e:
		traceback.print_exc()

# generator function for validation
def val_generator():
	try:
		# global variables
		global batch_size, input_shape, val_X, val_Y, len_val_X
		# getting length of validation data
		len_val_X = len(val_X)
		# converting labels into one-hot vector
		val_Y = np.array(pd.get_dummies(pd.Series(val_Y)))
		# runing generator
		while(True):
			for index in range(int(math.ceil(len_val_X/batch_size))):
				# making batch of data
				current_image_bucket = val_X[index * int(batch_size) : (index + 1) * int(batch_size), :, :, :]
				current_label_bucket = val_Y[index * int(batch_size) : (index + 1) * int(batch_size), :]
				yield current_image_bucket, current_label_bucket
	except Exception as e:
		traceback.print_exc()

# getting path of all vehicle and non-vehicle images
vehicle_images_path_list = get_data(vehicle_images_path)
non_vehicle_images_path_list = get_data(non_vehicle_images_path)
images_path_list = np.array(vehicle_images_path_list + non_vehicle_images_path_list)

# getting all images and labels
images, labels = read_images_make_labels(images_path_list)

# dividing data into train and validation
train_X, val_X, train_Y, val_Y = train_test_split(images, labels, test_size=0.1, stratify=labels, random_state=77)
len_train_X = len(train_X)
len_val_X = len(val_X)

# updating status
print("Train data: " + str(train_X.shape) + "\t" + str(train_Y.shape))
print("Val data: " + str(val_X.shape) + "\t" + str(val_Y.shape))

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

# training CNN model and saving it
model.fit_generator(train_generator(), epochs=epochs, steps_per_epoch=int(math.ceil(len_train_X/batch_size)), max_queue_size=batch_size, verbose=1)
model.save(cnn_model_path)

# loading pretrained model
#model.load_weights(cnn_model_path)

# validating trained model
pred = model.predict_generator(val_generator(), steps=int(math.ceil(len_val_X/batch_size)), max_queue_size=batch_size)
val_acc = accuracy_score(np.argmax(val_Y, 1), np.argmax(pred, 1))
val_kappa = cohen_kappa_score(np.argmax(val_Y, 1), np.argmax(pred, 1))
print("Val Acc: " + str(val_acc) + "\tVal Kappa: " + str(val_kappa))
