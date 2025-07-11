#!/usr/bin/env python
# coding: utf-8

# Library that Used in this Model

# In[ ]:


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
tf.random.set_seed(42)


# Get Train and Test Files from Kaggle and assgin Train file to variable (train) and Test file to variable (test)

# In[ ]:


#Read the CSV file (dataset)
train = pd.read_csv("Datasets/emnist-letters-train.csv", header=None)
test = pd.read_csv("Datasets/emnist-letters-test.csv", header=None)


# In[ ]:


train.head()


# In[ ]:


images = []
labels = []
for i in range(len(train)):
    image = np.flipud(np.rot90(train.iloc[i, 1:].to_numpy().reshape(28,28))) # iloc all Image Exclude the First
    images.append(image)
    label = train.iloc[i, 0] # First column only
    labels.append(label)
X_train = np.array(images)
y_train = np.array(labels)
y_train = y_train - 1


# In[ ]:


images = []
labels = []
for i in range(len(test)):
    image = np.flipud(np.rot90(test.iloc[i, 1:].to_numpy().reshape(28,28)))
    images.append(image)
    label = test.iloc[i, 0]
    labels.append(label)
X_test = np.array(images)
y_test = np.array(labels)
y_test = y_test - 1


# In[ ]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# are used to add an extra dimension to the input data arrays X_train and X_test. The purpose of adding this extra dimension is to make the data compatible with the input shape expected by certain types of neural network models, particularly when using libraries like TensorFlow or Keras.

# In[ ]:


X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# In[ ]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[ ]:


plt.figure(figsize = (12,10))
row, colums = 5, 5
for i in range(25):  
    plt.subplot(colums, row, i+1)
    plt.text(0.5, -2, chr(y_train[i] + 65))
    plt.imshow(X_train[i],interpolation='nearest', cmap='Greys')
plt.show()


#  ANN(Artificial Neural Network) Model

# In[ ]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512,  activation='relu'),
    keras.layers.Dense(256,  activation='relu'),
    keras.layers.Dense(128,  activation='relu'),
    keras.layers.Dense(64,   activation='relu'),
    keras.layers.Dense(26,   activation='softmax')
])


#  patience => The number of epochs with no improvement after which training will be stopped
#  In My case, it's set to 5, meaning that if the validation loss does not improve for 5 consecutive epochs, training will be stopped.
#  verbose => Controls the logging of messages. If set to 1, it will print a message when training is stopped due to early stopping.
#  mode => Specifies whether the monitored quantity should be minimized or maximized. In this case, it's set to 'min', indicating that training will stop when the quantity being monitored (validation loss) stops decreasing.

# In[ ]:


# Optimizing Algorithm for Backpropagation
optimizer_name = 'adam'
model.compile(
        optimizer=optimizer_name,
        loss='sparse_categorical_crossentropy',  # Multi classification
        metrics=['accuracy']
    )
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
mcp_save = ModelCheckpoint('ANN.model', save_best_only=True, monitor='val_loss', verbose=1, mode='auto')


# In[ ]:


history = model.fit(X_train,
                    y_train, 
                    epochs=20, 
                    batch_size=50, 
                    verbose=1, 
                    validation_split=0.1,
                    callbacks=[early_stopping, mcp_save])


# In[ ]:


val_loss, val_acc = model.evaluate(X_test, y_test)
print(f"val_loss is {val_loss}")
print(f"val_acc is {val_acc}")


# In[ ]:


def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)


# In[ ]:


plotgraph(epochs, acc, val_acc)


# In[ ]:


# loss curve
print('Loss Curve')
plotgraph(epochs, loss, val_loss)


# In[ ]:


plt.plot(history.history['accuracy'], color='b', label="accuracies")
plt.title("Train Accuracies")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


plt.plot(history.history['loss'], color='b', label="loss")
plt.title("Train loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


N=len(history.epoch) #epoch
plt.plot(np.arange(0,N),history.history['loss'],label='Training_loss')
plt.plot(np.arange(0,N),history.history['accuracy'],label='Accuracy')
plt.title('Training loss and accuracy')
plt.xlabel('Epochs')
plt.legend(loc='right')


# In[ ]:


# Save the model
model.save('ANN.model')


# In[ ]:


# Load best model
model = load_model("ANN.model")
model.summary()

