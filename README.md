# Convolutional Neural Network (CNN) for Fashion MNIST Classification

## Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify images from the **Fashion MNIST** dataset. It builds on a prior implementation using a Multilayer Perceptron (MLP), improving performance by leveraging CNNs' ability to retain spatial structure in images.

**Related MLP Project**: [tensorflow-mlp-fashion-mnist](https://github.com/adabyt/tensorflow-mlp-fashion-mnist)

---

## Dataset

The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset contains 70,000 grayscale images (28×28 pixels) of clothing across 10 categories:

- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Data split:
- **Training set**: 60,000 images  
- **Test set**: 10,000 images

---

## Preprocessing

- Normalised pixel values from 0–255 → 0.0–1.0
- Reshaped each image from (28, 28) → (28, 28, 1) to represent grayscale channels for Conv2D
- Used integer labels as-is for sparse categorical loss

---

## CNN Architecture

```text
Input: 28x28 grayscale image → reshaped to (28, 28, 1)

Conv2D: 32 filters (3×3), ReLU  
MaxPooling2D: pool size (2×2)

Conv2D: 64 filters (3×3), ReLU  
MaxPooling2D: pool size (2×2)

Flatten  
Dense: 128 neurons, ReLU  
Dense: 10 neurons, Softmax (for 10 classes)
```

- **Total parameters**: 225,034  
- **Loss**: Sparse Categorical Crossentropy  
- **Optimiser**: Adam  
- **Metrics**: Accuracy  
- **Training**: 10 epochs, batch size 32, validation split 10%

---

## Performance Summary

- **Training Accuracy**: Rose steadily to 95.95%
- **Validation Accuracy**: Peaked at **91.65%**
- **Test Accuracy**: **91.01%**
- **Test Loss**: **0.2874**

Compared to the [MLP](https://github.com/adabyt/tensorflow-mlp-fashion-mnist) (accuracy: 87.75%, loss: 0.3409), the CNN performs significantly better — as expected due to its ability to model spatial patterns in images.

---

## Observations

- **Epoch 7** may offer the best generalisation: it had the **lowest validation loss** (despite accuracy peaking later).
- CNN architecture enables better feature extraction by preserving image topology.

---

## Sample Predictions

```text
Sample 1: Predicted: Ankle boot | Actual: Ankle boot  
Sample 2: Predicted: Pullover   | Actual: Pullover  
Sample 3: Predicted: Trouser    | Actual: Trouser  
Sample 4: Predicted: Trouser    | Actual: Trouser  
Sample 5: Predicted: Shirt      | Actual: Shirt
```

---

## References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Keras Docs](https://www.tensorflow.org/guide/keras)
- [Original MLP Repo](https://github.com/adabyt/tensorflow-mlp-fashion-mnist)

---

## Author’s Note

This project directly follows the [MLP approach](https://github.com/adabyt/tensorflow-mlp-fashion-mnist) to demonstrate the power of convolutional architectures in image recognition tasks. Training metrics and architectural choices are included to serve as an educational baseline for future improvements.

---

## License

MIT License
