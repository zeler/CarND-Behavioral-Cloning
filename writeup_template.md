# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2021_04_15_21_40_57_491.jpg "Center example"
[image2]: ./examples/left_2021_04_15_21_56_41_882.jpg "Left example"
[image3]: ./examples/right_2021_04_27_10_04_04_051.jpg "Right example"
[image4]: ./examples/recovery1.jpg "Recovery Image"
[image5]: ./examples/recovery2.jpg "Recovery Image"
[image6]: ./examples/recovery3.jpg "Recovery Image"
[image7]: ./examples/history.png "History"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 with recorded drive around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The code used to implement the model and train and save the convolution neural network is stored in file 'Behavioral Cloning.ipynb'. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based on a neural network originally designed by NVIDIA to drive a real car. The model takes as an input an image with dimensions (160, 320, 3) and starts with 2 layers used for preprocessing of the data:

	- Cropping2D - crops the data to use only meaningful portion of the image which will make it easier for the network to train
	- Lambda - used for data normalization 

After these layers, 5 convolutional layers follow:

	- filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu'
	- filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu'
	- filters=48, kernel_size=(5, 5), strides=(2,2), activation='relu'
	- filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu'
	- filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu'

Then, a Flatten layers follows and 5 Dense layers with relu activation function, with following amounts of units:

	- 1164
	- 100
	- 50
	- 10
	- 1

The output of the final node is used as a steering angle for the car.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, I've added dropout layers after the Dense layers. The dropout rate is  0.25 for all layers. I've also used l2 regularization in convolutional layers with penalization value 0.0001. I've also experimented with BatchNormalization and l1-l2 regularization, however, these actually worsened the performance of the model. However, this kind of regularization might be more useful if I had a bigger dataset or if I aimed to run the car on multiple tracks. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. I used 20% of all data for validation purposes. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with initial learning rate set to 0.001. Higher learning rate converged faster, however, I got lower validation accuracy. During the training, I used batches of size 8. Bigger batches converge faster, however, I got better results with smaller batches. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As suggested in the curricullum, I used a combination of center lane driving, recovering from the left and right sides of the road and also driving in the opposite direction. Apart from the images from center camera, I also used the data from the left and right camera. I've also used image augmentation to increase even further the size of training dataset. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with known architectures, without trying to invent a completely new solution.

My first step was to use LENET convolution neural network, however I quickly realized the model isn't good eneough, as the loss was high even after a lot of training epochs. As suggested in the curriculum, I started to experiment with the CNN designed by NVIDIA researchers. The rationale behind this was that if this CNN is powerful enough to drive a car in real world, it should be also capable of driving car in a simulator. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The car was quickly capable of driving around the circuit, however, it didn't really generalize the model and it rather memorized it. This was obvious, as the MSE was very small for training set but a few orders of magnitude larger on the validation set. 

To combat the overfitting, I modified the model so that it used L2 regularization in the convolutional layers and dropout layer after the dense layers. I kept the dropout value rather small (0.25), as with higher values, the netowrk started to have difficulty to drive even on a straight part of the circuit. However, I had to experiment with different values for both L2 regularizer and dropouts to find the sweet spon in which the network had both training and validation MSE in acceptable intervals.  

The final step was to run the simulator to see how well the car was driving around track one. As expected, there were a few spots where the vehicle fell off the track - namely the final part of the circuit, once the car crosses the bridge. Since this part of the circuit was more complex than the rest (e.g. sharper turns or sudden turn to right side), I've added extra passes of this part to the dataset, using different car positions and approaching the turns from different sides of the road. I've also added extra samples of recovering the car from the side of the road, so it knew how to return back to the center of the lane.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

```python
dropout_rate = 0.25

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(filters=24, 
                 kernel_size=(5, 5), 
                 strides=(2,2), 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(filters=36, 
                 kernel_size=(5, 5), 
                 strides=(2,2), 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(filters=48, 
                 kernel_size=(5, 5), 
                 strides=(2,2), 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3), 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3), 
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.0001)))
model.add(Flatten())
model.add(Dense(units=1164, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(units=10, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded multiple laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I've also decided to use left and right camera images to increase the size of dataset:

![alt text][image2]
![alt text][image3] 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it drives to side of a road. These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

I also flipped images and angles, to give the CNN examples of turning to the other side of the road. The circuit contains mostly left turns, so the car runs into trouble if it needs to turn right. Apart from this, I also drived the circuit in the opposite direction to help the network with further generalization.

After the collection process, I had 9132 number of data points. By flipping all of avaialble images, I increased the size of the dataset to 18264 images. 

I preprocessed each image during the training in two steps:
	
- cropping the image using Cropping2D layer:
	
	Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))

- normalizing the image around zero using Lambda layer:

	Lambda(lambda x: (x / 255.0) - 0.5)

I also experimented with converting the image to grayscale, however, it didn't bring any noticeable result. This could be expected, as the network itself is quite powerful and usable in much more complex real-life scenarios. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. I didn't create and extra test set, as the network performance was tested using the simulator.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The network converged to optimal solution very quickly, after 5 epochs, as can be seen in the training history plot:

![alt text][image7]

I used an adam optimizer with the learing rate set to 0.001. To prevent overfitting, I used EarlyStopping callback along with ModelCheckpoint callback.
