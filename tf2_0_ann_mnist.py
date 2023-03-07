#import library
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Dataset
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data() #loads two tuple
x_train, x_test = x_train /255.0, x_test/255.0 #scale data in range 0-1


# Model creation and Fit
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

"""
#Note:
#flatten = N x 784
#128 is hyperparamater. Choose wisely
#dropout = 20% change for a neuron to be droppedout
#last layer = 10 outputs
"""

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=25)

#Plot loss and accuracy
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

# Evaluate and predict
model.evaluate(x_test,y_test)

p_test = model.predict(x_test).argmax(axis=1)

#lets show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0] #indices of wrong predicted images
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" %(y_test[i], p_test[i]))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, p_test)
