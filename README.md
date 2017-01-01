# sdc_behavior_cloning

### Background

This is 3rd project in Udacity Self-Driving Car Nano Degree. It requires 
collection of training data from udacity provided simulator to get the camera
images as seen from car's front alongwith the steering angle. A Convolutional
Neural network is then trained using data so that the network can predict 
steering angle based on the image. The trained network is tested again on
Simulator to run the car in autonomous mode on two tracks. 

### Data Collection
Collecting data from simulator turned out to be very tough as navigating the car 
on Simulator with keyboard produced data with lot of jumps in the steering angle.
In first pass I tried to use my data own data but it did not help the network 
learn well. I then switched over to udacity provided training data. 
This data is available at the following link 
[Udacity Data from Track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

In subsequent passes, while training the network, I just used the center camera 
images from udacity data and tried to augment with some more recordings
done on simulator but then ran into the same earlier issue of not getting clean 
smooth data using keyboard. Towards the later phases of project dropped that 
approach and decided to augment udacity provided data in different
ways as explained below:

1) __*Use of left and right camera images as augmented data*__: For using left 
camera image, adjusted the steering angle by adding an offset value of 0.25 to make 
the turning from left camera image a bit softer as compared to central camera. 
Similarly, for right camera images, adjusted the angle by subtracting an offset 
of 0.25 from steering angle as seen from central camera. 

2) __*Use of flipped images*__: Augmented data by adding Left-Right flipped 
version of central camera images alongwith -1.x steering angle. This also helps
in avoiding a bias in network to make more left turns since the training data had
more of left turn images. 

3) __*Reduction in images with zero steering value*__: Also reduced the number
of samples with zero steering to avoid the network from learning to constantly
produce zero steering as predicted values. I retained about 30% of the images
which has zero steering value.

4) __*Use of ImageDataGenerator*__: From Keras preprocessing module, used
 ImageDataGenerator to create images with some variations such as horizontal, 
 vertical, zoom and channel shifts of 10%. ImageDataGenerator also helped in 
 better memory management as training data was fed to the model using Keras
 `fit_generator` function in batches. 
 
### Training Approach
 
The data prepared as above was shuffled and was split into three sets 
a) Training b) Validation and c) Test set. Training and Validation sets were
used inside the `fit_generator` function. 
  
Adam optimizer with a learning rate of 1e-4 was used to train the model. The 
Theory behind Adam Optimizer can be found at [Adam: A Method for 
Stochastic Optimization](https://arxiv.org/abs/1412.6980v8).

The model was trained for 5 epochs.

Test Set was used to report the test set accuracy. This is over and above the visual
check of the model by running the trained model on Simulator. The car ran well
in Simulator using screen resolution of 640x480 with graphics quality of 
`Fastest`. I trained the model on a Mac Pro laptop and took about 10 minutes to 
train the model. 




### Model used

I first thought of using some of the pre trained models and started working
towards that. I then decided to use VGG but my model did not train well. At this point
I switched to Udacity provided data and also switched to a
network trained ground up using the NVIDIA architecture as provided in 
[this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The paper assumed input images of size 66x200 while the images from simulator 
was of size 160x320. Accordingly, The image collected for training from simulator was
resized to 66x200 and then fed into the network. 

The network used was close replica of NVIDIA and is explained below:

1) The input to network is of image sizes 66x200x3 in RGB format. 
2) First layer of network normalises the input values to range -1.0 to +1.0
3) This is followed with three layers of 2D Convolutional layer with filter
size of 5x5, stride of 2x2, border_mode of `valid` and ELU activation. ELU is
found to be very good in providing a faster convergence as explained in [this 
 paper](https://arxiv.org/pdf/1511.07289v1.pdf). Output of first convolutional
 layer has an output of size 31x98x24. 2nd layer's output is 14x47x36. Third
 layer's output size is 5x22x48.
4) This is followed by two more layers of 2D Convolutional layer with filter 
size of 3x3, stride of 1x1, padding type `valid` and ELU activation. Out of 
this 4th convolutional layer is 3x20x64 and that of 5th convolutional layer
is 1x18x64.
5) The data is then flattened and fed into a dropout layer with dropout rate 
 of 0.2 with again an ELU activation function. Use of dropout layer in this 
 and next fully connected layer helps avoid the problem of over-fitting.
6) This is followed by 4 fully connected layers of sizes 1164, 100, 50 and 10 
respectively. The first of these layers also have another dropout of 0.2. Each
fully connected layer uses ELU activation. 
7) The output from fully connected layer with size of 10 is fed into a single
neuron without any activation. This is the output of network predicting
steering angle. 

### Conclusions

The car runs fairly well in the simulator but has a long way to go from 
being used in a real life scenario. It also needs many more components 
apart from just driving based on camera images. I am hoping to learn these
other techniques in coming topics in the nano degree.

This project helped me dirty my hands with tensorflow, keras and also gain
understanding of practical challenges in training and designing networks, 
the importance of good quality data alongwith an appropriate deep learning 
network. 