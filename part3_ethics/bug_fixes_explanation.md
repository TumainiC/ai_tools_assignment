# MNIST TensorFlow Bug Fixes - Comprehensive Documentation

**Author:** AI Tools Assignment - Part 3  
**Date:** October 27, 2025  
**Purpose:** Educational resource for debugging TensorFlow/Keras models

---

## Overview

This document provides detailed explanations for each bug in the `buggy_mnist_model.py` script and their corresponding fixes in `fixed_mnist_model.py`. Each bug is categorized by type and includes:
- Bug description
- Why it's problematic
- How to fix it
- Code examples
- Common symptoms/error messages

---

## Bug Categories Summary

| Category | Count | Bugs |
|----------|-------|------|
| Dimension Mismatch | 3 | #2, #5, #6 |
| Loss Function Errors | 2 | #4, #8 |
| Data Preprocessing | 3 | #1, #2, #3 |
| Training Configuration | 3 | #9, #10, #11 |
| Other Subtle Bugs | 5 | #7, #12, #13, #14, #15, #16 |
| **Total** | **16** | |

---

## Detailed Bug Explanations

### 🔴 Category 1: Data Preprocessing Errors

#### Bug #1: Missing Data Normalization

**Location:** Line 25-26 (commented out)

**Buggy Code:**
```python
# Missing normalization
# X_train = X_train.astype('float32') / 255.0  # MISSING
# X_test = X_test.astype('float32') / 255.0    # MISSING
```

**Fixed Code:**
```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

**Why It's a Problem:**
- MNIST pixel values range from 0-255 (uint8)
- Neural networks train better with normalized inputs in [0, 1] or [-1, 1]
- Large input values can cause:
  - Gradient explosion/vanishing
  - Slow convergence
  - Activation saturation (especially with sigmoid/tanh)
  - Numerical instability

**Symptoms:**
- Very slow training
- Loss stays high or fluctuates wildly
- Accuracy stuck at ~10% (random guessing)
- NaN or Inf values in gradients

**How to Identify:**
```python
print(f"Data range: [{X_train.min()}, {X_train.max()}]")
# Before normalization: [0, 255]
# After normalization: [0.0, 1.0]
```

**Best Practices:**
- Always check data range before training
- Use consistent normalization for train/val/test
- Document normalization method for inference
- Consider standardization (mean=0, std=1) for some models

---

#### Bug #2: Incorrect Reshape (Missing Channel Dimension)

**Location:** Lines 29-30

**Buggy Code:**
```python
X_train = X_train.reshape(-1, 28, 28)  # BUG: Missing channel dimension
X_test = X_test.reshape(-1, 28, 28)    # BUG: Missing channel dimension
```

**Fixed Code:**
```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

**Why It's a Problem:**
- Conv2D layers expect 4D input: `(batch_size, height, width, channels)`
- MNIST images are grayscale (1 channel), but reshape omits this
- Tensorflow will interpret 3D input as `(batch_size, height, width)` with no channel
- This creates dimension mismatch with Conv2D layer

**Error Message:**
```
ValueError: Input 0 of layer "conv2d" is incompatible with the layer: 
expected min_ndim=4, found ndim=3. Full shape received: (None, 28, 28)
```

**How to Fix:**
- For grayscale images: add channel dimension of 1
- For RGB images: channel dimension is 3
- Always verify shape after reshape:
  ```python
  assert X_train.shape == (num_samples, height, width, channels)
  ```

**Visual Explanation:**
```
MNIST Original: (60000, 28, 28)       → 3D array
Buggy Reshape:  (60000, 28, 28)       → Still 3D (wrong!)
Fixed Reshape:  (60000, 28, 28, 1)    → 4D (correct!)
                 ↑      ↑   ↑   ↑
               samples  H   W  channels
```

---

#### Bug #3: Incorrect Train/Test Split

**Location:** Lines 36-40

**Buggy Code:**
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    train_size=0.8,      # BUG: Should use test_size
    random_state=42,
    shuffle=False        # BUG: Should shuffle
)
```

**Fixed Code:**
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2,       # 20% for validation
    random_state=42,
    shuffle=True,        # Shuffle data
    stratify=y_train     # Maintain class balance
)
```

**Why It's a Problem:**
1. **Using `train_size` instead of `test_size`:** Both work, but `test_size` is more conventional
2. **`shuffle=False`:** Critical issue!
   - MNIST data is ordered by class (0s, then 1s, then 2s, etc.)
   - Without shuffling, validation set might contain only certain classes
   - Model won't generalize well
   - Validation metrics will be misleading

3. **Missing `stratify`:**
   - Doesn't maintain class distribution in splits
   - Some classes might be over/underrepresented
   - Especially problematic with imbalanced datasets

**Symptoms:**
- Suspiciously high or low validation accuracy
- Validation loss doesn't correlate with training
- Model performs very differently on train vs. val
- Class imbalance issues

**Best Practices:**
```python
# Check class distribution before and after split
print("Original distribution:")
print(pd.Series(y_train).value_counts().sort_index())

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_train
)

print("\nTrain distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nValidation distribution:")
print(pd.Series(y_val).value_counts().sort_index())
```

---

### 🔴 Category 2: Dimension Mismatch Errors

#### Bug #5: Incorrect Input Shape in First Layer

**Location:** Line 58

**Buggy Code:**
```python
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28))  # BUG
```

**Fixed Code:**
```python
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
```

**Why It's a Problem:**
- Input shape must match data shape (excluding batch dimension)
- Missing channel dimension causes mismatch
- Conv2D requires `(height, width, channels)` format

**Error Message:**
```
ValueError: Input 0 of layer "sequential" is incompatible with the layer: 
expected shape=(None, 28, 28), found shape=(None, 28, 28, 1)
```

**Connection to Bug #2:**
This bug is directly related to Bug #2. If you fix Bug #2 (add channel dimension to data) but don't fix Bug #5, you get a dimension mismatch.

**Debugging Tips:**
```python
# Print model input shape
print(model.input_shape)  # Should be (None, 28, 28, 1)

# Print data shape
print(X_train.shape)      # Should be (samples, 28, 28, 1)

# They must match (except for batch dimension)
```

---

#### Bug #6: Missing Flatten Layer

**Location:** Line 62

**Buggy Code:**
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dense(64, activation='relu'),  # BUG: Can't connect 3D to 1D
    ...
])
```

**Fixed Code:**
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                    # FIX: Add Flatten layer
    layers.Dense(64, activation='relu'),
    ...
])
```

**Why It's a Problem:**
- Conv2D and MaxPooling2D output 3D tensors: `(batch, height, width, channels)`
- Dense layers expect 1D input: `(batch, features)`
- Can't directly connect 3D output to 1D input
- This is one of the most common beginner mistakes!

**Error Message:**
```
ValueError: Input 0 of layer "dense" is incompatible with the layer: 
expected min_ndim=2, found ndim=4. Full shape received: (None, 13, 13, 32)
```

**Understanding Tensor Flow:**
```
Input:           (batch, 28, 28, 1)
After Conv2D:    (batch, 26, 26, 32)     ← 3D (spatial)
After MaxPool:   (batch, 13, 13, 32)     ← 3D (spatial)
After Flatten:   (batch, 5408)           ← 1D (13*13*32=5408)
After Dense:     (batch, 64)             ← 1D (features)
Output:          (batch, 10)             ← 1D (classes)
```

**Alternative Solutions:**
```python
# Option 1: Flatten (most common)
layers.Flatten()

# Option 2: GlobalAveragePooling2D (reduces parameters)
layers.GlobalAveragePooling2D()

# Option 3: Reshape (manual)
layers.Reshape((-1,))
```

---

### 🔴 Category 3: Loss Function Errors

#### Bug #4: Missing One-Hot Encoding

**Location:** Lines 46-48 (commented out)

**Buggy Code:**
```python
# Missing one-hot encoding
# y_train = keras.utils.to_categorical(y_train, 10)  # MISSING
# y_val = keras.utils.to_categorical(y_val, 10)      # MISSING
# y_test = keras.utils.to_categorical(y_test, 10)    # MISSING
```

**The Issue:**
This bug is related to Bug #8 (wrong loss function). There are two valid approaches:

**Approach A: Integer Labels + Sparse Loss**
```python
# Labels stay as integers: [0, 1, 2, ..., 9]
# Use sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Approach B: One-Hot Labels + Categorical Loss**
```python
# Convert labels to one-hot: [[1,0,0,...], [0,1,0,...], ...]
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Why It Matters:**
- Mismatch between label format and loss function causes errors
- `categorical_crossentropy` expects one-hot encoded labels
- `sparse_categorical_crossentropy` expects integer labels
- Using wrong combination will cause error or incorrect training

**Error with Mismatch:**
```python
# If labels are integers but using categorical_crossentropy:
ValueError: Shapes (None, 1) and (None, 10) are incompatible

# If labels are one-hot but using sparse_categorical_crossentropy:
InvalidArgumentError: Received a label value of X which is outside 
the valid range of [0, 10)
```

**Label Format Comparison:**
```python
# Original integer format
y_train[0]  # Output: 5

# One-hot encoded format
y_train[0]  # Output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#                               ↑ (5th index = 1)
```

**Which to Use?**
- **Integer labels (sparse):** More memory efficient, simpler
- **One-hot labels (categorical):** Needed for some advanced techniques (label smoothing)
- **Recommendation:** Use `sparse_categorical_crossentropy` for most cases

---

#### Bug #8: Wrong Loss Function

**Location:** Line 78

**Buggy Code:**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # BUG: Wrong for multi-class
    metrics=['accuracy']
)
```

**Fixed Code:**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Correct for integer labels
    metrics=['accuracy']
)
```

**Why It's a Problem:**
- **Binary crossentropy:** For binary classification (2 classes: 0 or 1)
- **Categorical crossentropy:** For multi-class (10 classes: 0-9) with one-hot labels
- **Sparse categorical crossentropy:** For multi-class with integer labels
- MNIST is 10-class problem, so binary_crossentropy is completely wrong

**How to Choose Loss Function:**

| Problem Type | Label Format | Loss Function |
|--------------|--------------|---------------|
| Binary (2 classes) | Integer (0, 1) | `binary_crossentropy` |
| Binary (2 classes) | One-hot ([1,0], [0,1]) | `binary_crossentropy` |
| Multi-class | Integer (0-9) | `sparse_categorical_crossentropy` |
| Multi-class | One-hot ([1,0,0...]) | `categorical_crossentropy` |
| Multi-label | Binary array ([1,0,1...]) | `binary_crossentropy` |
| Regression | Continuous values | `mse`, `mae` |

**Symptoms of Wrong Loss:**
- Model accuracy stuck at random guess level (~10% for 10 classes)
- Loss doesn't decrease
- Validation accuracy fluctuates wildly
- Predictions are nonsensical

**Validation Check:**
```python
# Check output layer and loss compatibility
output_layer = model.layers[-1]
print(f"Output units: {output_layer.units}")
print(f"Output activation: {output_layer.activation.__name__}")
print(f"Loss function: {model.loss}")

# For 10-class classification:
# Output units: 10
# Output activation: softmax
# Loss function: sparse_categorical_crossentropy (or categorical_crossentropy)
```

---

### 🔴 Category 4: Training Configuration Errors

#### Bug #9: Variable Name Typo + Bug #10: Learning Rate Too High

**Location:** Line 90

**Buggy Code:**
```python
optimzer = keras.optimizers.Adam(learning_rate=10.0)  # Two bugs!
#  ↑ Typo                                     ↑ Too high

model.compile(
    optimizer=optimzer,  # BUG: Typo
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Fixed Code:**
```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Correct name & value

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Bug #9 - Variable Name Typo:**
- Python won't catch this until runtime
- If variable is defined but never used, no error occurs
- If you try to use it later, get `NameError`
- Common typos: `optimzer`, `optimizr`, `optimiser`

**Bug #10 - Learning Rate Too High:**
- Default Adam LR is 0.001 (1e-3)
- LR of 10.0 is 10,000x too large!
- Causes training instability and divergence

**Effects of High Learning Rate:**
```
Learning Rate: 10.0
Epoch 1: loss: nan - accuracy: 0.1000
Epoch 2: loss: nan - accuracy: 0.1000
...
(Training fails completely)
```

**Learning Rate Guidelines:**

| Optimizer | Typical Range | Default | Notes |
|-----------|--------------|---------|-------|
| SGD | 0.01 - 0.1 | 0.01 | Needs momentum |
| Adam | 0.0001 - 0.01 | 0.001 | Adaptive, robust |
| RMSprop | 0.0001 - 0.01 | 0.001 | Good for RNNs |
| Adagrad | 0.001 - 0.1 | 0.01 | Adapts over time |

**How to Find Good LR:**
```python
# LR Range Test (Learning Rate Finder)
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):
    return lr * 1.1  # Increase by 10% each epoch

lr_callback = LearningRateScheduler(lr_schedule)

# Train and plot loss vs. learning rate
# Choose LR just before loss starts increasing
```

---

#### Bug #11: Incorrect Validation Data Format

**Location:** Line 104

**Buggy Code:**
```python
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=X_val,  # BUG: Missing y_val
    verbose=1
)
```

**Fixed Code:**
```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),  # FIX: Tuple of (X, y)
    verbose=1
)
```

**Why It's a Problem:**
- `validation_data` expects tuple: `(X_validation, y_validation)`
- Providing only `X_val` causes error
- Model needs both features and labels to compute validation loss

**Error Message:**
```
TypeError: `validation_data` should be a tuple 
`(val_x, val_y, val_sample_weight)` or `(val_x, val_y)`. Found: ...
```

**Alternative Validation Methods:**
```python
# Method 1: Tuple (most common)
validation_data=(X_val, y_val)

# Method 2: Validation split (automatic)
validation_split=0.2  # Use 20% of training data

# Method 3: Dataset object
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_data=val_dataset
```

**Important Notes:**
- Don't use both `validation_split` and `validation_data`
- `validation_split` takes from END of training data
- Always shuffle data before using `validation_split`

---

### 🔴 Category 5: Other Subtle Bugs

#### Bug #7: Wrong Activation Function

**Location:** Line 66

**Buggy Code:**
```python
layers.Dense(10, activation='sigmoid')  # BUG: Wrong for multi-class
```

**Fixed Code:**
```python
layers.Dense(10, activation='softmax')  # Correct for multi-class
```

**Why It's a Problem:**
- **Sigmoid:** Outputs values in [0, 1], each output is independent
  - Good for: Binary classification, multi-label classification
  - Each output represents P(class=1)
  
- **Softmax:** Outputs sum to 1, represents probability distribution
  - Good for: Multi-class classification
  - Each output represents P(class=i) where sum = 1

**For MNIST:**
- 10 classes (digits 0-9)
- Each sample belongs to exactly ONE class
- Need probabilities that sum to 1
- **Must use softmax**

**Visual Comparison:**
```python
# Sigmoid output example (WRONG for MNIST):
[0.8, 0.3, 0.9, 0.1, 0.5, 0.2, 0.7, 0.4, 0.6, 0.3]
# Sum: 4.8 (makes no sense as probabilities)

# Softmax output example (CORRECT for MNIST):
[0.05, 0.02, 0.65, 0.01, 0.10, 0.01, 0.08, 0.02, 0.05, 0.01]
# Sum: 1.0 (valid probability distribution)
```

**Activation Function Guide:**

| Use Case | Output Layer Activation | Loss Function |
|----------|------------------------|---------------|
| Binary Classification | sigmoid | binary_crossentropy |
| Multi-Class (exclusive) | softmax | categorical/sparse_categorical |
| Multi-Label | sigmoid | binary_crossentropy |
| Regression | linear (none) | mse, mae |

---

#### Bug #12: Using Wrong Dataset in Evaluation

**Location:** Line 113

**Buggy Code:**
```python
test_loss, test_accuracy = model.evaluate(X_train, y_test)  # BUG: X_train!
```

**Fixed Code:**
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

**Why It's a Problem:**
- Evaluating on training data gives overly optimistic results
- Doesn't measure generalization ability
- Defeats the purpose of having separate test set
- Classic "data leakage" mistake

**Impact:**
```python
# Typical results:
Train accuracy: 99.5%   # Model has memorized training data
Test accuracy:  97.8%   # True generalization performance

# If you evaluate on training data:
"Test" accuracy: 99.5%  # Misleading! Not actual test performance
```

**Best Practices:**
```python
# 1. Never use test set during development
# Use validation set for hyperparameter tuning

# 2. Only evaluate on test set once at the end
test_loss, test_acc = model.evaluate(X_test, y_test)

# 3. Document which dataset was used
print("Evaluation on test set:")
print(f"Samples: {len(X_test)}")
print(f"Accuracy: {test_acc:.4f}")
```

---

#### Bug #13: Missing argmax for Predictions

**Location:** Lines 119-121

**Buggy Code:**
```python
predictions = model.predict(X_test[:5])
print("\nFirst 5 predictions:")
print(predictions)  # BUG: Prints probabilities, not class labels
```

**Output (confusing):**
```
[[0.01, 0.05, 0.02, 0.03, 0.01, 0.65, 0.08, 0.10, 0.03, 0.02],
 [0.85, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.03, 0.01, 0.02],
 ...]
```

**Fixed Code:**
```python
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)

print("\nFirst 5 predictions:")
for i, (pred, pred_class, actual) in enumerate(zip(predictions, predicted_classes, y_test[:5])):
    confidence = pred[pred_class]
    print(f"Sample {i+1}:")
    print(f"  Predicted: {pred_class} (confidence: {confidence:.4f})")
    print(f"  Actual: {actual}")
```

**Output (clear):**
```
Sample 1:
  Predicted: 5 (confidence: 0.6500)
  Actual: 5
Sample 2:
  Predicted: 0 (confidence: 0.8500)
  Actual: 0
...
```

**Understanding Predictions:**
```python
# model.predict() returns probabilities
predictions = model.predict(X_test)
# Shape: (num_samples, num_classes)
# Example: predictions[0] = [0.01, 0.02, ..., 0.65, ..., 0.01]

# To get class labels, use argmax
predicted_classes = np.argmax(predictions, axis=1)
# Shape: (num_samples,)
# Example: predicted_classes[0] = 5

# To get confidence, index the probability array
confidence = predictions[0][predicted_classes[0]]
# or
confidence = np.max(predictions[0])
```

---

#### Bug #14: Incorrect Manual Accuracy Calculation

**Location:** Lines 124-129

**Buggy Code:**
```python
correct = 0
for i in range(len(X_test)):
    pred = predictions[i]  # BUG: predictions only has 5 samples!
    if pred == y_test[i]:  # BUG: Comparing array to integer
        correct += 1

manual_accuracy = correct / len(y_test)  # Will crash
```

**Multiple Issues:**
1. `predictions` only has 5 samples (from earlier line)
2. Attempting to loop over all test samples
3. Comparing probability array to integer label
4. Index out of bounds error

**Error Message:**
```
IndexError: index 5 is out of bounds for axis 0 with size 5
```

**Fixed Code:**
```python
# Get predictions for full test set
all_predictions = model.predict(X_test)
predicted_classes = np.argmax(all_predictions, axis=1)

# Calculate accuracy correctly
correct = np.sum(predicted_classes == y_test)
manual_accuracy = correct / len(y_test)

print(f"Manual accuracy: {manual_accuracy:.4f}")
```

**Efficient Alternatives:**
```python
# Method 1: NumPy (fastest)
accuracy = np.mean(predicted_classes == y_test)

# Method 2: scikit-learn
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_classes)

# Method 3: Manual (educational)
correct = sum(pred == true for pred, true in zip(predicted_classes, y_test))
accuracy = correct / len(y_test)
```

---

#### Bug #15: Wrong History Dictionary Keys

**Location:** Lines 139-140

**Buggy Code:**
```python
plt.plot(history.history['acc'])      # BUG: Wrong key
plt.plot(history.history['val_acc'])  # BUG: Wrong key
```

**Fixed Code:**
```python
plt.plot(history.history['accuracy'])      # Correct
plt.plot(history.history['val_accuracy'])  # Correct
```

**Why It's a Problem:**
- TensorFlow 1.x used keys: `'acc'`, `'val_acc'`
- TensorFlow 2.x uses keys: `'accuracy'`, `'val_accuracy'`
- Using old keys causes `KeyError`
- Common issue when using old tutorials/code

**Error Message:**
```
KeyError: 'acc'
```

**Available History Keys:**
```python
print("Available keys:", history.history.keys())
# Output: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# Access correctly:
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
```

**Version-Agnostic Code:**
```python
# Check which keys exist
if 'accuracy' in history.history:
    acc_key = 'accuracy'
    val_acc_key = 'val_accuracy'
elif 'acc' in history.history:
    acc_key = 'acc'
    val_acc_key = 'val_acc'
else:
    raise KeyError("Could not find accuracy metrics")

plt.plot(history.history[acc_key])
plt.plot(history.history[val_acc_key])
```

---

#### Bug #16: Wrong Model File Extension

**Location:** Line 156

**Buggy Code:**
```python
model.save('mnist_model.model')  # BUG: .model is not standard
```

**Fixed Code:**
```python
# Option 1: HDF5 format (compatible)
model.save('mnist_model.h5')

# Option 2: Native Keras format (recommended)
model.save('mnist_model.keras')

# Option 3: SavedModel format (TF serving)
model.save('saved_model/')
```

**Why It Matters:**
- `.model` is not a recognized format
- May load incorrectly or fail
- Different formats have different features
- Important for deployment and compatibility

**Model Format Comparison:**

| Format | Extension | Pros | Cons | Use Case |
|--------|-----------|------|------|----------|
| HDF5 | `.h5` | Compact, widely supported | Single file only | Development, sharing |
| Keras | `.keras` | Native, complete | TF 2.x only | Recommended for TF 2.x |
| SavedModel | `/directory` | Production-ready, versioning | Larger size | Deployment, TF Serving |
| JSON + Weights | `.json` + `.h5` | Separate architecture/weights | Manual loading | Custom workflows |

**Saving and Loading Examples:**
```python
# Save model
model.save('my_model.h5')

# Load model
from tensorflow import keras
loaded_model = keras.models.load_model('my_model.h5')

# Verify loaded model
loaded_model.summary()
test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model accuracy: {test_acc:.4f}")
```

**SavedModel Format (Production):**
```python
# Save
model.save('saved_model/')

# Directory structure:
# saved_model/
#   ├── assets/
#   ├── variables/
#   │   ├── variables.data-00000-of-00001
#   │   └── variables.index
#   └── saved_model.pb

# Load
loaded_model = keras.models.load_model('saved_model/')
```

---

## Summary of All Fixes

### Quick Reference Table

| Bug # | Category | Issue | Fix |
|-------|----------|-------|-----|
| 1 | Preprocessing | No normalization | Add `/255.0` |
| 2 | Dimensions | Missing channel | Reshape to `(28,28,1)` |
| 3 | Preprocessing | Wrong split params | Use `test_size`, `shuffle=True` |
| 4 | Loss Function | Label encoding | Use sparse loss OR one-hot |
| 5 | Dimensions | Wrong input shape | Add channel to `input_shape` |
| 6 | Dimensions | Missing Flatten | Add `Flatten()` layer |
| 7 | Activation | Sigmoid vs Softmax | Use `softmax` for multi-class |
| 8 | Loss Function | Binary vs Categorical | Use `sparse_categorical_crossentropy` |
| 9 | Configuration | Variable typo | Fix spelling: `optimizer` |
| 10 | Configuration | LR too high | Use `0.001` not `10.0` |
| 11 | Configuration | Validation format | Use tuple `(X_val, y_val)` |
| 12 | Evaluation | Wrong dataset | Use `X_test` not `X_train` |
| 13 | Prediction | No argmax | Add `np.argmax(predictions, axis=1)` |
| 14 | Calculation | Logic error | Fix loop and comparison |
| 15 | Visualization | Wrong keys | Use `'accuracy'` not `'acc'` |
| 16 | Saving | Wrong extension | Use `.h5` or `.keras` |

---

## Testing and Validation

### How to Verify Each Fix

```python
# 1. Data Normalization
assert X_train.min() >= 0.0 and X_train.max() <= 1.0, "Data not normalized"

# 2. Shape Check
assert X_train.shape == (num_samples, 28, 28, 1), "Wrong shape"

# 3. Split Check
assert len(X_train) + len(X_val) == original_size, "Split error"

# 4-8. Model Compilation
assert model.built, "Model not built"
assert len(model.layers) > 0, "No layers"

# 9-11. Training Check
assert 'accuracy' in history.history, "Training failed"
assert len(history.history['accuracy']) > 0, "No epochs run"

# 12. Evaluation Check
assert 0 <= test_accuracy <= 1, "Invalid accuracy"

# 13-14. Prediction Check
assert predicted_classes.shape == y_test.shape, "Shape mismatch"

# 15. History Keys
assert 'accuracy' in history.history.keys(), "Wrong keys"

# 16. Model Save/Load
loaded = keras.models.load_model('mnist_model.h5')
assert loaded is not None, "Failed to load"
```

---

## Common Error Messages and Solutions

### Quick Diagnostic Guide

| Error Message | Likely Bug | Quick Fix |
|---------------|-----------|-----------|
| `expected min_ndim=4, found ndim=3` | Bug #2 or #5 | Add channel dimension |
| `expected min_ndim=2, found ndim=4` | Bug #6 | Add Flatten layer |
| `Shapes (None, 1) and (None, 10) are incompatible` | Bug #4 or #8 | Fix label encoding or loss |
| `NameError: name 'optimzer' is not defined` | Bug #9 | Fix typo |
| `Loss is NaN` | Bug #10 | Reduce learning rate |
| `validation_data should be a tuple` | Bug #11 | Use `(X_val, y_val)` |
| `KeyError: 'acc'` | Bug #15 | Use `'accuracy'` |
| `IndexError: index ... out of bounds` | Bug #14 | Fix array access |

---

## Learning Outcomes

After working through these bugs, students should understand:

1. **Data Pipeline:** Normalization, reshaping, splitting
2. **Model Architecture:** Input shapes, layer compatibility, dimensions
3. **Loss Functions:** Choosing appropriate loss for problem type
4. **Training Process:** Compilation, fitting, validation
5. **Evaluation:** Using correct datasets, interpreting predictions
6. **Debugging Skills:** Reading error messages, checking shapes, validating data
7. **Best Practices:** Code organization, variable naming, documentation

---

## Additional Resources

### Recommended Reading
1. [TensorFlow Official Documentation](https://www.tensorflow.org/api_docs)
2. [Keras API Reference](https://keras.io/api/)
3. [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet

### Debugging Tools
```python
# Print all tensor shapes in model
for layer in model.layers:
    print(f"{layer.name}: {layer.output_shape}")

# Check data statistics
print(f"Min: {X_train.min()}, Max: {X_train.max()}")
print(f"Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")

# Verify label distribution
import pandas as pd
print(pd.Series(y_train).value_counts().sort_index())

# Monitor training with callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

---

## Conclusion

These 16 bugs represent common mistakes made by beginners and even experienced developers when working with TensorFlow and Keras. Understanding why each bug occurs and how to fix it is crucial for:

- **Debugging:** Quickly identifying and resolving issues
- **Best Practices:** Writing robust, maintainable code
- **Deep Learning:** Understanding model architecture and training
- **Production:** Deploying reliable, well-tested models

Remember: **Most bugs can be prevented by:**
1. Checking shapes at every step
2. Validating data ranges and formats
3. Reading error messages carefully
4. Testing small changes incrementally
5. Using type hints and documentation

---

**Happy Debugging! 🐛 → 🦋**
