import tensorflow as tf
from tensorflow import keras
from keras import layers 
from keras import models 

import numpy as np
import matplotlib.pyplot as plt

print("\n----- 1. Loading and Initial Data Preparation -----")

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Original x_train shape: {x_train.shape}")   # (60000, 28, 28)
print(f"Original y_train shape: {y_train.shape}")   # (60000,)
print(f"Original x_test shape: {x_test.shape}")     # (10000, 28, 28)
print(f"Original y_test shape: {y_test.shape}")     # (10000,)

# Define class names for better understanding
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("-"*100)

print("\n----- 2. Data Preprocessing for CNNs -----")

# Instead of flattening as with the MLP, we need to add a "channels" dimension
    # Normalisation: Same as before (0-255 to 0.0-1.0)
    # Reshaping: For a grayscale image, Conv2D layers expect input shape to be (height, width, channels). 
        # Since Fashion MNIST images are grayscale, they have 1 channel. So, a 28x28 image needs to become 28x28x1.

# Convert integer pixel values to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalise pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"\nx_train shape after dtype conversion and normalisation: {x_train.shape}") # (60000, 28, 28)
print(f"x_test shape after dtype conversion and normalisation: {x_test.shape}")     # (10000, 28, 28)

# Reshape the data to add the channels dimension
# For grayscale, channels=1
# For color (RGB), channels=3
# Input shape for Conv2D should be (height, width, channels)
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"x_train_cnn shape after reshaping for CNN: {x_train_cnn.shape}")            # (60000, 28, 28, 1)
print(f"x_test_cnn shape after reshaping for CNN: {x_test_cnn.shape}")              # (10000, 28, 28, 1)

# y_train and y_test are already integer labels (0-9), which is suitable for SparseCategoricalCrossentropy

print("-"*100)

print("\n----- 3. Model Definition: Building the CNN Architecture -----")

# Define the Keras Sequential model for CNN
model_cnn = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(
        filters=32,                         # The number of filters (feature detectors) you want the layer to learn. This determines the depth of the output feature map.
        kernel_size=(3, 3),                 # The dimensions of the filter. Each filter will be 3x3 pixels..
        activation='relu', 
        input_shape=(28, 28, 1)             # Crucial for the first layer only, specifying the shape of a single input image.
        ),
    layers.MaxPooling2D(pool_size=(2, 2)),  # Halves the spatial dimensions (e.g., 28x28 -> 14x14)

    # Second Convolutional Block
    layers.Conv2D(
        filters=64,                         # It will learn 64 feature maps in the second convolutional layer.
        kernel_size=(3, 3), 
        activation='relu'
        ),
    layers.MaxPooling2D(pool_size=(2, 2)),  # Halves again (e.g., 14x14 -> 7x7)

    # Flatten the output to feed into Dense layers
    layers.Flatten(),

    # Dense layers for classification (same as MLP's head)
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

model_cnn.summary()

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩ 
# │ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │  # input_shape=(28, 28, 1); kernel_size=(3,3), filters=32; Total parameters = (kernel_height * kernel_width * input_channels + 1 (bias)) * number_of_filters = (3 * 3 * 1 + 1) * 32 = (9 + 1) * 32 = 10 * 32 = 320
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │  # Total parameters = (kernel_height * kernel_width * input_channels + 1) * number_of_filters = (3 * 3 * 32 + 1) * 64 = (288 + 1) * 64 = 289 * 64 = 18496
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 1600)                │               0 │  # The Flatten layer reshapes the data from (5, 5, 64) to 5 * 5 * 64 = 1600 
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │         204,928 │  # Total parameters = (input_features * neurons) + neurons = (1600 * 128) + 128 = 204800 + 128 = 204928
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │  # Total parameters = (input_features * neurons) + neurons = (128 * 10) + 10 = 1280 + 10 = 1290
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 225,034 (879.04 KB)
#  Trainable params: 225,034 (879.04 KB)
#  Non-trainable params: 0 (0.00 B)

print("-"*100)

print("\n----- 4. Model Compilation -----")

# Same as the MLP
# Compile the CNN model
model_cnn.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

print("-"*100)

print("\n----- 5. Model Training -----")

# Using the CNN-reshaped data (x_train_cnn, x_test_cnn) for training and evaluation

# Train the CNN model
print("\nStarting CNN training...")
history_cnn = model_cnn.fit(
    x_train_cnn, y_train,
    epochs=10, 
    batch_size=32,
    validation_split=0.1
    )

# Starting CNN training...
# Epoch 1/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.7710 - loss: 0.6380 - val_accuracy: 0.8667 - val_loss: 0.3662
# Epoch 2/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.8809 - loss: 0.3222 - val_accuracy: 0.8920 - val_loss: 0.2960
# Epoch 3/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9024 - loss: 0.2652 - val_accuracy: 0.8917 - val_loss: 0.2886
# Epoch 4/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9134 - loss: 0.2335 - val_accuracy: 0.9028 - val_loss: 0.2656
# Epoch 5/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9216 - loss: 0.2073 - val_accuracy: 0.9052 - val_loss: 0.2608
# Epoch 6/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9334 - loss: 0.1820 - val_accuracy: 0.9098 - val_loss: 0.2564
# Epoch 7/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9390 - loss: 0.1629 - val_accuracy: 0.9118 - val_loss: 0.2540
# Epoch 8/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9467 - loss: 0.1432 - val_accuracy: 0.9095 - val_loss: 0.2601
# Epoch 9/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9552 - loss: 0.1230 - val_accuracy: 0.9165 - val_loss: 0.2590
# Epoch 10/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9595 - loss: 0.1081 - val_accuracy: 0.9155 - val_loss: 0.2599


print("-"*100)

print("\n----- 6. Model Evaluation -----")

# Evaluate the CNN model on the test data
test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(x_test_cnn, y_test, verbose=1)

print(f"\nCNN Test Loss: {test_loss_cnn:.4f}")          # 0.2874
print(f"CNN Test Accuracy: {test_accuracy_cnn:.4f}")    # 0.9101

# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9095 - loss: 0.2945 

# CNN Test Loss: 0.2874
# CNN Test Accuracy: 0.9101

print("-"*100)

print("\n----- 7. Making Predictions -----")

# Make predictions on a few test samples using the CNN model
predictions_cnn = model_cnn.predict(x_test_cnn[:5])
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)

print(f"\nCNN Predictions for the first 5 test samples (classes): {predicted_classes_cnn}") # [9 2 1 1 6]
print(f"Actual classes for the first 5 samples: {y_test[:5]}")                              # [9 2 1 1 6]

print("\nCNN Comparison of predicted vs actual:")
for i in range(5):
    print(f"Sample {i+1}: Predicted: {class_names[predicted_classes_cnn[i]]} (Index: {predicted_classes_cnn[i]}), Actual: {class_names[y_test[i]]} (Index: {y_test[i]})")

# CNN Comparison of predicted vs actual:
# Sample 1: Predicted: Ankle boot (Index: 9), Actual: Ankle boot (Index: 9)
# Sample 2: Predicted: Pullover (Index: 2), Actual: Pullover (Index: 2)
# Sample 3: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 4: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 5: Predicted: Shirt (Index: 6), Actual: Shirt (Index: 6)
