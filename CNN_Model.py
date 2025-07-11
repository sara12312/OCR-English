import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
tf.random.set_seed(42)

# Read the CSV file (dataset)
train = pd.read_csv(r"Datasets/emnist-letters-train.csv", header=None)
test = pd.read_csv(r"Datasets/emnist-letters-test.csv", header=None)

# Preprocess data for CNN
def preprocess_data(data):
    images = []
    labels = []
    for i in range(len(data)):
        image = np.flipud(np.rot90(data.iloc[i, 1:].to_numpy().reshape(28, 28)))
        images.append(image)
        label = data.iloc[i, 0]
        labels.append(label)
    X = np.array(images)
    y = np.array(labels)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    y = y - 1
    return X, y

X_train, y_train = preprocess_data(train)
X_test, y_test = preprocess_data(test)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build CNN model
model = keras.models.Sequential([
    # feature learning
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # machine learning 
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5), # prevent overfitting
    keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('CNN.model', save_best_only=True)

# Train the CNN model
history = model.fit(
    X_train, y_train,
    epochs=1,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the CNN model on the test set
val_loss, val_acc = model.evaluate(X_test, y_test)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
