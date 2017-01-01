from keras.models import Sequential
from keras.layers import ELU, Lambda
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import ImageOps
from sklearn.model_selection import train_test_split
import json
import pandas as pd

# Motivated from the Nvidia Paper
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

def get_model():
    r, c, ch = 66, 200, 3 # rows, columns and channels

    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(r, c, ch)))
    # input size is 66, 200, 3
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    # output shape is 31, 98, 24
    # Activation function used is ELU as it is supposed to give faster convergence to learning
    # https://arxiv.org/pdf/1511.07289v1.pdf
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    # output shape is 14, 47, 36
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    # output shape is 5, 22, 48
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    # output shape is 3, 20, 64
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    # output shape is 1, 18, 64
    model.add(Flatten())
    # dropout layers to avoid overfitting
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(1164))
    # dropout layers to avoid overfitting
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))
    # the model has one single output. No sigmoid is applied. Loss function is mse.
    # THis is to get the output steering angle as close as possible to the provided value during training

    return model


# Method to load image form file. Accepts "mirror" parameter to generate a flipped image.
def load_image(img_path, mirror=False):
    img = image.load_img("./data/" + img_path.strip(), target_size=(66, 200))
    if (mirror):
        img = ImageOps.mirror(img)
    x = image.img_to_array(img)
    return x

# Method to load images from paths
def load_images(paths, mirror=False):
    return np.array([load_image(path, mirror) for path in paths])


if __name__ == "__main__":

    print("Start of Training")

    # Read data from log file to a pandas data frame
    data = pd.read_csv('./data/driving_log.csv')

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # rows with zero steering
    data_zero = data[data.steering == 0]

    # rows with non zero steering
    data_curved = data[data.steering != 0]

    # retain 30% of rows with steering = 0
    # Most of the data is with steering = 0 and to balance the data more and
    # teach the network to provide non zero values for bends and recovery from sides
    # we need to phave more data from those situations
    data_zero_s = data_zero.sample(frac=0.30).reset_index(drop=True)


    # merge back data
    data_trucn = pd.concat([data_zero_s, data_curved]).reset_index(drop=True)

    # shuffle the data once more
    data_trucn = data_trucn.sample(frac=1).reset_index(drop=True)

    # get steering angles as float32
    steering_angles = data_trucn.steering.values.astype(np.float32)

    # get center, left and right image paths
    center_image_paths = data_trucn.center.values
    left_images_paths = data_trucn.left.values
    right_images_paths = data_trucn.right.values

    # Steering coefficient to add or subtract from left and right training images
    steering_coefficient = 0.25

    # Generate steering angles - center, left, right and center mirrored
    # this is first step of data augmentation
    # more augmentation is done using Image generator
    steering_angles_mirror = steering_angles * -1.
    steering_angles_left = steering_angles + steering_coefficient
    steering_angles_right = steering_angles - steering_coefficient

    steering_angles = np.concatenate((steering_angles, steering_angles_mirror))
    steering_angles = np.concatenate((steering_angles, steering_angles_left))
    steering_angles = np.concatenate((steering_angles, steering_angles_right))

    # Load center, left, right and center mirrored images
    images = load_images(center_image_paths)
    images = np.concatenate((images, load_images(center_image_paths, mirror=True)))
    images = np.concatenate((images, load_images(left_images_paths)))
    images = np.concatenate((images, load_images(right_images_paths)))

    # At this point all images are in `images` with corresponding angles in `steering_angles`

    # Split data for training and validation+test
    images_train, images_val_test, \
    steering_angles_train, steering_angles_val_test = \
        train_test_split(images,
                        steering_angles,
                        test_size=0.2,
                        random_state=100)

    # Split validation+test dataset into validation and test sets
    images_test, images_val, steering_angles_test, \
    steering_angles_val = train_test_split(images_val_test,
                                            steering_angles_val_test,
                                            test_size=0.5,
                                            random_state=100)

    # Load the model
    model = get_model()

    # save model
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    # Compile model
    model.compile(loss='mse',
                  optimizer=Adam(lr=1e-4))

    # Create a generator that would shift weights, heights and channels channels, and also zoom.
    image_generator = ImageDataGenerator(width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         zoom_range=0.1,
                                         channel_shift_range=0.1,
                                         fill_mode='nearest')

    # Define number of epochs
    nb_epoch = 5

    # Train the model using generator
    model.fit_generator(image_generator.flow(images_train, steering_angles_train, batch_size=64),
                        samples_per_epoch=len(steering_angles_train),
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=image_generator.flow(images_val, steering_angles_val),
                        nb_val_samples=len(steering_angles_val))

    # Test the model
    loss_test = model.evaluate(images_test, steering_angles_test)
    print("Loss on test data:", loss_test)

    # Save weights
    model.save_weights('model.h5')

    print("End of Training")

