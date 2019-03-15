
import numpy as np

import os
import glob

import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.layers import ZeroPadding2D, Flatten, Dropout
from keras.optimizers import Adagrad
from time import time
import keras as k
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint


def train_generator(path,speed):
    image_size = (384, 512)
    batch_size = 4
    image_batch = np.zeros((batch_size, 512, 384, 6))
    label_batch = np.zeros((batch_size))
    idx=0
    while True:
        for i in range(batch_size):
            #print(path[idx])
            #print(type(path[idx]))

            image_1 = cv2.imread(path[idx])
            image_2 = cv2.imread(path[idx+1])

            image_resized_1 = cv2.resize(image_1, image_size)
            image_resized_2 = cv2.resize(image_2, image_size)
            concat_image = np.concatenate((image_resized_1, image_resized_2), axis=-1)
            #print(concat_image.shape)
            #concat_speed = np.mean([float(speed[idx+1].rstrip()),float(speed[idx].rstrip())])
            #concat_speed = np.concatenate(float(speed[idx+1].rstrip()),float(speed[idx].rstrip()))
            concat_speed = np.append([float(speed[idx].rstrip())], [float(speed[idx+1].rstrip())])
            concat_speed = np.reshape(concat_speed, (1,2))
            idx += 1
            #plt.imshow(image_1)
            #plt.show()
            image_batch[i]= concat_image
            label_batch[i]= concat_speed
        yield shuffle(image_batch/255., label_batch)

def flowNet():

    conv1 = k.layers.ZeroPadding2D((3, 3))(model_input)
    conv1 = k.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='valid', name='conv1')(conv1)
    conv1 = k.layers.BatchNormalization()(conv1)
    conv1 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu1')(conv1)

    conv2 = k.layers.ZeroPadding2D((2, 2))(conv1)
    conv2 = k.layers.Conv2D(128, kernel_size=(5, 5), strides=2, padding='valid', name='conv2')(conv2)
    conv2 = k.layers.BatchNormalization()(conv2)
    conv2 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu2')(conv2)

    conv3a = k.layers.ZeroPadding2D((2, 2))(conv2)
    conv3a = k.layers.Conv2D(256, kernel_size=(5, 5), strides=2, padding='valid', name='conv3a')(conv3a)
    conv3a = k.layers.BatchNormalization()(conv3a)
    conv3a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3a')(conv3a)

    conv3b = k.layers.ZeroPadding2D()(conv3a)
    conv3b = k.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='valid', name='conv3b')(conv3b)
    conv3b = k.layers.BatchNormalization()(conv3b)
    conv3b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu3b')(conv3b)

    conv4 = k.layers.ZeroPadding2D()(conv3b)
    conv4 = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv4a')(conv4)
    conv4 = k.layers.BatchNormalization()(conv4)
    conv4 = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4')(conv4)

    conv4b = k.layers.ZeroPadding2D()(conv4)
    conv4b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv4b')(conv4b)
    conv4b = k.layers.BatchNormalization()(conv4b)
    conv4b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu4b')(conv4b)

    conv5a = k.layers.ZeroPadding2D()(conv4b)
    conv5a = k.layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding='valid', name='conv5a')(conv5a)
    conv5a = k.layers.BatchNormalization()(conv5a)
    conv5a = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5a')(conv5a)

    conv5b = k.layers.ZeroPadding2D()(conv5a)
    conv5b = k.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='valid', name='conv5b')(conv5b)
    conv5b = k.layers.BatchNormalization()(conv5b)
    conv5b = k.layers.Lambda(lambda x: k.layers.activations.relu(x), name='relu5b')(conv5b)

    conv6 = k.layers.ZeroPadding2D()(conv5b)
    conv6 = k.layers.Conv2D(1024, kernel_size=(3, 3), strides=2, padding='valid', name='conv6')(conv6)
    conv6 = k.layers.BatchNormalization()(conv6)
    conv6 = k.layers.MaxPool2D(pool_size=(2, 3))(conv6)  # only difference in addition from original paper

    flat = k.layers.Flatten()(conv6)

    fc6 = k.layers.Dense(1024, activation='relu', name='fc6')(flat)
    fc6_dropout = k.layers.Dropout(self.dropout)(fc6)

    fc7 = k.layers.Dense(1024, activation='relu', name='fc7')(fc6_dropout)
    fc7_dropout = k.layers.Dropout(self.dropout)(fc7)

    category_output = k.layers.Dense(self.num_buckets, activation='softmax', name='category')(fc7_dropout)
    speed_output = k.layers.Lambda(lambda x: x, name='speed')(category_output)

    return model


def main():

   DATA_PATH = os.getcwd()

   F = open('train.txt', 'r')
   # print(F.read())
   ground_truth = F.read().splitlines()
   # print(len(ground_truth))
   paths = []
   for x in sorted(glob.glob("Data/*.jpg")):
       paths.append(x)
   # print(len(paths))

   #data_train, data_val, labels_train, labels_val = train_test_split(paths, ground_truth, test_size=0.30)
   data_train, labels_train = paths[0:15000], ground_truth[0:15000]
   data_val, labels_val = paths[15000:], ground_truth[15000:]
   dataset = train_generator(data_train, labels_train)
   X,Y = next(dataset)
   validation = train_generator(data_val, labels_val)
   X,Y = next(validation)
   model = flowNet()
   print(model.summary())
   tensorboard= TensorBoard(log_dir="logs/{}".format(time()))
   checkpointer = ModelCheckpoint("model_checkpoint.hdf5", monitor='val_loss', save_best_only=True, save_weights_only=True)

   #model.compile(optimizer=Adagrad(lr=0.001),
   #              loss='mse', metrics=['accuracy'])
   #model.fit_generator(generator=dataset, steps_per_epoch=250, epochs=50, verbose=1,
    #                   validation_data=validation, validation_steps=10,callbacks=[tensorboard, checkpointer])
   model.save("Final_model.h5")
main()
