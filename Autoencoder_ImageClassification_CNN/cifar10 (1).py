import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization
from matplotlib import pyplot
import numpy as np
import pickle
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
from keras.datasets import cifar10
import random
from keras.utils import to_categorical
from keras.models import Sequential, Model

#loading data from cifar10 dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()

print('Train: X = %s, y = %s' %(np.shape(trainX), np.shape(trainY)))
print('Test: Y = %s, y = %s' %(np.shape(testX), np.shape(testY)))

#didnt end up using the class names, was going to add it to the printout of the images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#####printing 5 random images from the dataset
r = list(range(10000))
random.shuffle(r)

for i in range(5):
    num = random.choice(r)
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(trainX[num])

pyplot.show()

for i in range(5):
    num = random.choice(r)
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(testX[num])

pyplot.show()
#####end random image printer

#changing the pixel data to make it easier to use
trainX, testX = trainX / 255.0, testX / 255.0

#didnt use this
#encoding values to transform the integer into a 10 element binary
#vector with a 1 for the index of the class value
# trainY = to_categorical(trainY)
# testY = to_categorical(testY)

#didnt use
#prepare the pixel data
# def pixels(train, test):
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     return train_norm, test_norm


#########starting constructing the CNN Layers
#first 2 layer model with max pooling
model = Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
#1 layer
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
#2 layer
#model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

#second 2 level model with average pooling
model2 = Sequential()
model2.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model2.add(layers.AveragePooling2D((2,2)))

model2.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model2.add(layers.AveragePooling2D((2,2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10))
model2.summary()
#model2.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

#3rd model with L = 3 and max pooling
model3 = Sequential()
model3.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model3.add(layers.MaxPooling2D((2,2)))
#1 layer
model3.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model3.add(layers.MaxPooling2D((2,2)))
#2 layer
model3.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model3.add(layers.MaxPooling2D((2,2)))
#layer 3 
model3.add(layers.Flatten())
model3.add(layers.Dense(64, activation='relu'))
model3.add(layers.Dense(10))
model3.summary()

#4th model with L = 3 and avg pooling
model4 = Sequential()
model4.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model4.add(layers.AveragePooling2D((2,2)))
#1 layer
model4.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model4.add(layers.AveragePooling2D((2,2)))
#2 layer
model4.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model4.add(layers.AveragePooling2D((2,2)))
#layer 3 
model4.add(layers.Flatten())
model4.add(layers.Dense(64, activation='relu'))
model4.add(layers.Dense(10))
model4.summary()

#######I cant get the versions with 4 layers to work 
# #5th model with L = 4 and avg pooling
# model5 = Sequential()
# model5.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
# model5.add(layers.AveragePooling2D((2,2)))
# #1 layer
# model5.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model5.add(layers.AveragePooling2D((2,2)))
# #2 layer
# model5.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model5.add(layers.AveragePooling2D((2,2)))
# #layer 3 
# model5.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model5.add(layers.AveragePooling2D((2,2)))
# model5.add(layers.Conv2D(64, (3,3), activation = 'relu'))

# model5.add(layers.Flatten())
# model5.add(layers.Dense(64, activation='relu'))
# model5.add(layers.Dense(10))
# model5.summary()

# #6th model with L = 4 and max pooling
# model6 = Sequential()
# model6.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
# model6.add(layers.MaxPooling2D((2,2)))
# #1 layer
# model6.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model6.add(layers.MaxPooling2D((2,2)))
# #2 layer
# model6.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model6.add(layers.MaxPooling2D((2,2)))
# #layer 3 
# model6.add(layers.Conv2D(64, (3,3), activation = 'relu'))
# model6.add(layers.MaxPooling2D((2,2)))
# model5.add(layers.Conv2D(64, (3,3), activation = 'relu'))

# model6.add(layers.Flatten())
# model6.add(layers.Dense(64, activation='relu'))
# model6.add(layers.Dense(10))
# model6.summary()
##############end model creation section


######Model Fitting Section, Commented out because it takes so long to compile
# model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
# #(trainX, trainY), (testX, testY) = cifar10.load_data()

# history = model.fit(trainX, trainY, epochs = 10, validation_data = (testX, testY))
# pyplot.plot(history.history['accuracy'], label = 'Accuracy Model 1- 2L Max Pool')
# pyplot.plot(history.history['val_accuracy'], label = 'val_accuary Model 1 - 2L Max Pool')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Accuracy')
# pyplot.legend(loc = 'lower right')
# pyplot.show()

# test_loss, test_acc = model.evaluate(testX, testY, verbose =2)
# print('Test 1 loss = ', test_loss, 'Test 1 accuracy = ', test_acc)


# model2.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
# #(trainX, trainY), (testX, testY) = cifar10.load_data()

# history = model2.fit(trainX, trainY, epochs = 10, validation_data = (testX, testY))
# pyplot.plot(history.history['accuracy'], label = 'Accuracy Model 2 - 2L AVG Pool')
# pyplot.plot(history.history['val_accuracy'], label = 'val_accuary Model 2 - 2L AVG Pool')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Accuracy')
# pyplot.legend(loc = 'lower right')
# pyplot.show()

# test_loss, test_acc = model2.evaluate(testX, testY, verbose =2)
# print('Test 2 loss = ', test_loss, 'Test 2 accuracy = ', test_acc)


# model3.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
# #(trainX, trainY), (testX, testY) = cifar10.load_data()

# history = model3.fit(trainX, trainY, epochs = 10, validation_data = (testX, testY))
# pyplot.plot(history.history['accuracy'], label = 'Accuracy Model 3 - 3L Max Pool')
# pyplot.plot(history.history['val_accuracy'], label = 'val_accuary Model 3 - 3L Max Pool')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Accuracy')
# pyplot.legend(loc = 'lower right')
# pyplot.show()

# test_loss, test_acc = model3.evaluate(testX, testY, verbose =2)
# print('Test 3 loss = ', test_loss, 'Test 3 accuracy = ', test_acc)


# model4.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
# #(trainX, trainY), (testX, testY) = cifar10.load_data()

# history = model4.fit(trainX, trainY, epochs = 10, validation_data = (testX, testY))
# pyplot.plot(history.history['accuracy'], label = 'Accuracy Model 4 - 3L AVG Pool')
# pyplot.plot(history.history['val_accuracy'], label = 'val_accuary Model 4 - 3L AVG Pool')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Accuracy')
# pyplot.legend(loc = 'lower right')
# pyplot.show()

# test_loss, test_acc = model4.evaluate(testX, testY, verbose =2)
# print('Test 4 loss = ', test_loss, 'Test 4 accuracy = ', test_acc)
######end model compile and print

######Autoencoder part of the homework
input_img = Input(shape=(32,32,3))
model5 = Sequential()
model5.add(Conv2D(32, kernel_size=3, strides=1, padding='same',activation = 'relu', input_shape = (32, 32, 3)))

model5.add(BatchNormalization())     # 32x32x32
model5.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
model5.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
model5.add(BatchNormalization())     # 16x16x32
model5.add(UpSampling2D())
model5.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
model5.add(BatchNormalization())
model5.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

model5.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
model5.summary()

#Encoder
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

#Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
###commenting out because this took 2 hours to compile
# model5.fit(trainX, trainX,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(testX, testX))

# predicted = model5.predict(testX)

# pyplot.figure(figsize=(40,4))
# for i in range(10):
#     # display original images
#     ax = pyplot.subplot(3, 20, i + 1)
#     pyplot.imshow(testX[i].reshape(32, 32,3))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
    
#     # display reconstructed images
#     ax = pyplot.subplot(3, 20, 2*20 +i+ 1)
#     pyplot.imshow(predicted[i].reshape(32, 32,3))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
 
# pyplot.show()

#creating a smaller model, fit with 10 epochs
history = model5.fit(trainX, trainX,
                 epochs=10,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(testX, testX))

#printing plots for loss vs val_loss and accuracy vs val acc.
pyplot.plot(history.history['accuracy'], label = 'Accuracy Model 5 -Autoencoder')
pyplot.plot(history.history['val_accuracy'], label = 'val_accuary Model 5 - Autoencoder')
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.legend(loc = 'lower right')
pyplot.show()

pyplot.plot(history.history['loss'], label = 'Loss Model 5 -Autoencoder')
pyplot.plot(history.history['val_loss'], label = 'val_loss Model 5 - Autoencoder')
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.legend(loc = 'lower right')
pyplot.show()


test_loss, test_acc = model5.evaluate(testX, testY, verbose =2)
print('Test 5 loss = ', test_loss, 'Test 5 accuracy = ', test_acc)
#end file, see report for overview