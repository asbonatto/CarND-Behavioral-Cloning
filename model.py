"""
Self Driving Car Nanodegree
Project : Behavioral Cloning

"""

import argparse
from data_feeder import DataFeeder
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Cropping2D, Lambda, BatchNormalization, Input
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


def enhanced_conv2d(model, filters, kernel_size, strides = (1,1)):
    """
    Custom layer that perform convolution followed by batch
    normalization.
    """
    model.add(Conv2D(filters, kernel_size, strides=strides, padding='valid', activation = "linear"))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(BatchNormalization())

    return model


def nvidia_model():
    """
    Implements the NVIDIA model as published in 
    https://arxiv.org/pdf/1604.07316
    """
    model = Sequential()
    input_shape = (160, 320, 3)
    # Normalization layer
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
    # Cropping layer : target is 66x200x3
    model.add(Cropping2D(cropping=((74, 20), (60, 60))))
    # Start of the convolutional stack
    for filter in [24, 36, 48]:
        model = enhanced_conv2d(model, filter, 5, strides=(2, 2))
        
    for filter in [64, 64]:
        model = enhanced_conv2d(model, filter, 3)
    
    # The original model does not contain dropout layer
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    for neurons in [100, 50, 10]:
        model.add(Dense(neurons, activation = 'linear'))
        model.add(LeakyReLU(alpha = 0.01))
        
 
    model.add(Dense(1))
    
    return model
    
def train_network(model, epochs = 5):
    """
    Main script for training the Behavioral Cloning 
    Network model
    """
    modelname = "model"
    print(model.summary())
    
    data = DataFeeder()
    
    callbacks = [ModelCheckpoint('model{epoch:02d}.h5')]
    #callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-6, verbose=1))

    model.compile(optimizer = "adam", loss = "mse")
    history = model.fit_generator(data.fetch_train(), nb_epoch = epochs, steps_per_epoch = data.steps_per_epoch, \
                                  validation_data = data.fetch_valid(), validation_steps = data.validation_steps, \
                                  verbose = 1, callbacks = callbacks)
    
    model.save(modelname + ".h5")
    print("Model saved to {}.h5".format(modelname))
    
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(modelname + '_training_history.png')
    plt.close(fig)
    
    return "Finish"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", help = "# of epochs for training")
    args = parser.parse_args()
    train_network(nvidia_model(), epochs = int(args.epochs))



    
    