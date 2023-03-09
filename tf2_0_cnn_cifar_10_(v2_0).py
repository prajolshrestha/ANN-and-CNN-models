import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten,MaxPooling2D, Dropout,BatchNormalization,GlobalMaxPooling2D

data = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = data.load_data()



#Noramlize input image pixel value such that they are in between 0-1
x_train, x_test = x_train/ 255.0, x_test/ 255.0

#Flatten labels
y_train,y_test = y_train.flatten(), y_test.flatten()

#unique labels
k = len(set(y_train))

###Create Model

#input_layer
i = Input(shape=x_train[0].shape)

#first hidden layer
x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

#second hidden layer
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

#third hidden layer
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

#Flatten
x = GlobalMaxPooling2D()(x)
#x = Flatten()(x)
x = Dropout(0.2)(x)

#Dense layer
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(k,activation='softmax')(x)

model = Model(i,x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=(['accuracy']))

#r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 50)

# Fit with data augmentation
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                                 height_shift_range=0.1,
                                                                 horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)

steps_per_epoch = x_train.shape[0] // batch_size
r = model.fit(train_generator, validation_data=(x_test,y_test), steps_per_epoch=steps_per_epoch, epochs=50)

#Plot loss and accuracy
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history('accuracy'), label='accuracy')
plt.plot(r.history('val_accuracy'), label='val_accuracy')
plt.legend()

model.evaluate(x_test,y_test)

p_test = model.predict(x_test).argmax(axis=1)

#confusion_matrix
cm = tf.math.confusion_matrix(x_test,p_test)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.y_label('True Value')

labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()

import numpy as np
#show misclassified examples
misclassified_idx = np.where(p_test !=  y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True Label: %s Predicted: %s" %(labels[y_test[i]], labels[p_test[i]]));

model.summary()
