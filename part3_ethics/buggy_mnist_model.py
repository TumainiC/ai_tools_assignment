"""
Buggy MNIST Classification Script
This code contains intentional errors for educational purposes.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ==================== SECTION 1: DATA LOADING ====================
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# BUG #1: Missing data preprocessing - no normalization
# Images should be normalized to [0, 1] range
# X_train = X_train.astype('float32') / 255.0  # MISSING
# X_test = X_test.astype('float32') / 255.0    # MISSING

# BUG #2: Incorrect reshaping - wrong dimensions
# Should reshape to (samples, 28, 28, 1) for Conv2D input
X_train = X_train.reshape(-1, 28, 28)  # BUG: Missing channel dimension
X_test = X_test.reshape(-1, 28, 28)    # BUG: Missing channel dimension

print(f"\nReshaped training data: {X_train.shape}")
print(f"Reshaped test data: {X_test.shape}")

# BUG #3: Incorrect train/validation split
# Using wrong parameter names and incorrect stratification
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    train_size=0.8,      # BUG: Should use test_size instead
    random_state=42,
    shuffle=False        # BUG: Should shuffle for better training
)

print(f"\nAfter split:")
print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# BUG #4: Wrong label encoding for categorical crossentropy
# Labels should be one-hot encoded for categorical_crossentropy
# y_train = keras.utils.to_categorical(y_train, 10)  # MISSING
# y_val = keras.utils.to_categorical(y_val, 10)      # MISSING
# y_test = keras.utils.to_categorical(y_test, 10)    # MISSING

print(f"Label shape: {y_train.shape}")  # Should be (samples, 10) after one-hot encoding

# ==================== SECTION 2: MODEL ARCHITECTURE ====================

# BUG #5: Incorrect input shape in first layer
# Input shape should be (28, 28, 1) not (28, 28)
model = keras.Sequential([
    # BUG: Wrong input shape - missing channel dimension
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),  # BUG HERE
    layers.MaxPooling2D((2, 2)),
    
    # BUG #6: Dimension mismatch - incompatible layer connection
    # After Conv2D and MaxPooling, we have 3D tensor, but Dense expects 1D
    layers.Dense(64, activation='relu'),  # BUG: Missing Flatten layer before this
    
    layers.Dropout(0.5),
    
    # BUG #7: Wrong activation function for multi-class classification
    # Should use 'softmax' for multi-class, not 'sigmoid'
    layers.Dense(10, activation='sigmoid')  # BUG: Wrong activation
])

print("\nModel architecture created")

# ==================== SECTION 3: MODEL COMPILATION ====================

# BUG #8: Wrong loss function for the problem
# For one-hot encoded labels, should use 'categorical_crossentropy'
# For integer labels, should use 'sparse_categorical_crossentropy'
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # BUG: Wrong loss function
    metrics=['accuracy']
)

# BUG #9: Missing model.compile() - commented out
# Model must be compiled before training
# This is already done above, but students might uncomment this section
# and find the bug below

print("\nModel compiled")

# Display model summary
model.summary()

# ==================== SECTION 4: TRAINING CONFIGURATION ====================

# BUG #10: Incompatible optimizer configuration
# Learning rate too high, might cause divergence
optimzer = keras.optimizers.Adam(learning_rate=10.0)  # BUG: Typo in variable name + lr too high

# BUG #11: Attempting to recompile with buggy optimizer
# Also, variable name is misspelled
model.compile(
    optimizer=optimzer,  # BUG: Typo - should be 'optimizer'
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==================== SECTION 5: MODEL TRAINING ====================

# BUG #12: Incorrect validation data format
# validation_data should be a tuple (X_val, y_val)
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=X_val,  # BUG: Missing y_val - should be (X_val, y_val)
    verbose=1
)

print("\nTraining completed")

# ==================== SECTION 6: EVALUATION ====================

# BUG #13: Wrong variable name in evaluation
# Using 'X_train' instead of 'X_test'
test_loss, test_accuracy = model.evaluate(X_train, y_test)  # BUG: Should use X_test

print(f"\nTest accuracy: {test_accuracy:.4f}")

# ==================== SECTION 7: PREDICTIONS ====================

# BUG #14: Incorrect prediction interpretation
# For categorical output, need to use argmax to get class labels
predictions = model.predict(X_test[:5])
print("\nFirst 5 predictions:")
print(predictions)  # BUG: Should use np.argmax to get class labels

# BUG #15: Incorrect metric calculation
# Manually calculating accuracy with wrong logic
correct = 0
for i in range(len(X_test)):
    pred = predictions[i]  # BUG: predictions array doesn't exist for full test set
    if pred == y_test[i]:
        correct += 1

manual_accuracy = correct / len(y_test)  # BUG: This will fail
print(f"Manual accuracy: {manual_accuracy:.4f}")

# ==================== SECTION 8: VISUALIZATION ====================

# BUG #16: Accessing wrong keys in history
# Keys should be 'accuracy' and 'val_accuracy', not 'acc' and 'val_acc'
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])      # BUG: Wrong key name
plt.plot(history.history['val_acc'])  # BUG: Wrong key name
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('training_history.png')
print("\nTraining history saved to 'training_history.png'")

# ==================== SECTION 9: SAVE MODEL ====================

# BUG #17: Incorrect file extension and path
# Should save as .h5 or .keras, not .model
model.save('mnist_model.model')  # BUG: Wrong extension
print("Model saved successfully")

print("\n" + "="*60)
print("SCRIPT COMPLETED - But with many bugs!")
print("="*60)

"""
INSTRUCTOR NOTES - Bug Summary:
================================
1. Missing data normalization
2. Incorrect reshape (missing channel dimension)
3. Wrong train_test_split parameters
4. Missing one-hot encoding for labels
5. Wrong input shape in first layer
6. Missing Flatten layer before Dense
7. Wrong activation function (sigmoid instead of softmax)
8. Wrong loss function (binary_crossentropy instead of categorical)
9. Variable name typo (optimzer)
10. Learning rate too high (10.0)
11. Wrong validation_data format
12. Using X_train instead of X_test in evaluation
13. Missing argmax for predictions
14. Incorrect manual accuracy calculation
15. Wrong history keys (acc instead of accuracy)
16. Wrong model file extension

Total: 16+ distinct bugs covering all required categories
"""
