"""
Self Driving Car Nanodegree
Project : Behavioral Cloning

"""

import pandas as pd
import numpy as np
import os
import cv2
import sklearn

import argparse

from keras.models import Model
from keras.layers import Cropping2D, Lambda, BatchNormalization, Input
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D


def read_drv_log(filename = "data/driving_log.csv", newpath = "IMG/"):
    """
    Returns the driving log in pandas dataframe format
    Image paths are converted to relative paths
    """
    
    df = pd.read_csv(filename)
    df.to_csv(filename + ".bkp", index = False)
    
    for col in ["right", "center", "left"]:
        for token in ["\\", "/"]:
            df[col] = df[col].str.split(token).str[-1]
            
        df[col] = df[col].apply(lambda x: os.path.join(newpath, x))

    # Saving the file with the correct data path. Used to 
    # clean up the file structure to run on multiple
    # computers
    df.to_csv(filename, index = False)
    
    # Timestamping the lines so I can tag multiple data collection campaigns
    col = "center"
    token = "_"
    df["timestamp"] = df[col].str.split(token).str[1:-2]
    df["timestamp"] = df["timestamp"].apply(lambda x: "".join(x))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format = "%Y%m%d%H%M")
    
    # Finally tagging the data collection campaigns
    df["campaign"] = df['timestamp'].diff().fillna(10*60*1E9).astype("timedelta64[m]")
    df["campaign"] = df["campaign"].apply(lambda x: 1 if (x > 2) else 0)
    df["campaign"] = df["campaign"].cumsum()
    
    return df

def feed_data(samples, batch_size = 128):
    """
    Generator function to feed data to the model without
    exhausting the computer memory.
    OBS:
        Just small changes from the generator lesson
    """
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.loc[offset:offset+batch_size]

            images = []
            angles = []
            for _, batch_sample in batch_samples.iterrows():
                center_image = cv2.imread("data/" + batch_sample["center"])
                center_angle = float(batch_sample["steering"])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def enhanced_conv2d(layer, filters, kernel_size, strides = (1,1)):
    """
    Custom layer that perform convolution followed by batch
    normalization.
    """
    layer = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer)
    layer = BatchNormalization()(layer)    
    return layer


def nvidia_model():
    """
    Implements the NVIDIA model as published in 
    https://arxiv.org/pdf/1604.07316
    """
    
    inputs = Input(shape = (160, 320, 3))
    # Normalization layer
    layer = Lambda(lambda x: x / 255.0 - 0.5)(inputs)
    layer = Cropping2D(cropping=((70, 25), (0, 0)))(layer)
    
    layer = enhanced_conv2d(layer, 24, 5, strides=(2, 2))
    layer = enhanced_conv2d(layer, 36, 5, strides=(2, 2))
    layer = enhanced_conv2d(layer, 48, 5, strides=(2, 2))
    layer = enhanced_conv2d(layer, 64, 3)
    layer = enhanced_conv2d(layer, 64, 3)
    
    # The original model does not contain dropout layer
    layer = Flatten()(layer)
    layer = Dense(100)(layer)
    layer = Dense( 50)(layer)
    layer = Dense( 10)(layer)
    layer = Dense(  1)(layer)
    model = Model(inputs=inputs, outputs=layer)
    
    return model
    
def train_network(model, epochs = 5):
    """
    Main script for training the Behavioral Cloning 
    Network model
    """
    samples = read_drv_log()
    BATCH_SIZE = 128
    steps_per_epoch = len(samples) // BATCH_SIZE
    
    model.compile(optimizer = "adam", loss = "mse")
    model.fit_generator(feed_data(samples, batch_size = BATCH_SIZE), nb_epoch = epochs, verbose = 1, steps_per_epoch = steps_per_epoch)
    model.save("model.h5")
    return "Finish"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", help = "# of epochs for training")
    args = parser.parse_args()
    train_network(nvidia_model(), epochs = int(args.epochs))



    
    