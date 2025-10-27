"""
Fixed MNIST Classification Script
This code corrects all bugs from the buggy version.
Each fix is documented with detailed comments.
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

# FIX #1: Add proper data normalization
# Normalize pixel values from [0, 255] to [0, 1] for better training
# This helps with gradient descent convergence and prevents saturation
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print("\n✓ FIX #1: Data normalized to [0, 1] range")

# FIX #2: Correct reshaping with channel dimension
# Conv2D expects 4D input: (samples, height, width, channels)
# MNIST is grayscale, so channels = 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print("✓ FIX #2: Data reshaped to include channel dimension")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")

# FIX #3: Correct train/validation split
# Use test_size parameter and enable shuffling for better training
# Stratify by labels to maintain class distribution
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2,      # FIX: Use test_size (20% for validation)
    random_state=42,
    shuffle=True,       # FIX: Enable shuffling
    stratify=y_train    # FIX: Maintain class balance in both sets
)
print("✓ FIX #3: Correct train/validation split with shuffling")
print(f"  Train: {X_train.shape}, Validation: {X_val.shape}")

# FIX #4: Proper label encoding
# Two options for multi-class classification:
# Option A: One-hot encoding with categorical_crossentropy
# Option B: Keep integer labels with sparse_categorical_crossentropy
# We'll use Option B (sparse) as it's more memory-efficient

# No need to one-hot encode when using sparse_categorical_crossentropy
# Labels remain as integers [0-9]
print("✓ FIX #4: Using integer labels with sparse_categorical_crossentropy")
print(f"  Label shape: {y_train.shape} (integer format)")
print(f"  Label range: [{y_train.min()}, {y_train.max()}]")

# Alternative Fix #4: If using one-hot encoding
# Uncomment these lines if you want to use categorical_crossentropy instead
"""
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(f"Labels one-hot encoded: {y_train.shape}")
"""

# ==================== SECTION 2: MODEL ARCHITECTURE ====================

print("\n" + "="*60)
print("BUILDING MODEL ARCHITECTURE")
print("="*60)

# FIX #5: Correct input shape with channel dimension
# FIX #6: Add Flatten layer before Dense layers
# FIX #7: Use softmax activation for multi-class classification
model = keras.Sequential([
    # FIX #5: Correct input shape (28, 28, 1)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Add another conv layer for better feature extraction
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # FIX #6: Add Flatten layer to convert 3D tensor to 1D
    # This is crucial: Conv2D outputs 3D, but Dense expects 1D
    layers.Flatten(),
    
    # Now we can use Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    
    # FIX #7: Use softmax for multi-class classification
    # Softmax ensures output probabilities sum to 1
    layers.Dense(10, activation='softmax')
])

print("\n✓ FIX #5: Correct input shape (28, 28, 1)")
print("✓ FIX #6: Added Flatten layer before Dense layers")
print("✓ FIX #7: Using softmax activation for output layer")

# Display model summary
print("\nModel Summary:")
model.summary()

# ==================== SECTION 3: MODEL COMPILATION ====================

print("\n" + "="*60)
print("COMPILING MODEL")
print("="*60)

# FIX #8: Use correct loss function
# sparse_categorical_crossentropy for integer labels
# categorical_crossentropy for one-hot encoded labels
# binary_crossentropy only for binary classification

# FIX #9: Correct variable name (optimizer, not optimzer)
# FIX #10: Use appropriate learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Default: 0.001

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # FIX #8: Correct loss function
    metrics=['accuracy']
)

print("✓ FIX #8: Using sparse_categorical_crossentropy loss")
print("✓ FIX #9: Fixed variable name typo (optimizer)")
print("✓ FIX #10: Using appropriate learning rate (0.001)")

# Alternative Fix #8: If using one-hot encoded labels
"""
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Use this for one-hot encoded labels
    metrics=['accuracy']
)
"""

# ==================== SECTION 4: TRAINING ====================

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# FIX #11: Correct validation_data format
# Must be a tuple: (X_val, y_val)
history = model.fit(
    X_train, y_train,
    epochs=10,          # Increased epochs for better training
    batch_size=128,     # Larger batch size for faster training
    validation_data=(X_val, y_val),  # FIX #11: Correct tuple format
    verbose=1
)

print("\n✓ FIX #11: Correct validation_data format (X_val, y_val)")
print("Training completed successfully!")

# ==================== SECTION 5: EVALUATION ====================

print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

# FIX #12: Use correct test data (X_test, not X_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"✓ FIX #12: Using correct test data (X_test)")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ==================== SECTION 6: PREDICTIONS ====================

print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

# FIX #13: Correctly interpret predictions
# Get predictions for first 5 test samples
predictions = model.predict(X_test[:5], verbose=0)

print("✓ FIX #13: Correct prediction interpretation")
print("\nFirst 5 predictions (probabilities):")
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    actual_class = y_test[i]
    
    print(f"Sample {i+1}:")
    print(f"  Predicted: {predicted_class} (confidence: {confidence:.4f})")
    print(f"  Actual: {actual_class}")
    print(f"  Correct: {'✓' if predicted_class == actual_class else '✗'}")

# FIX #14: Correct manual accuracy calculation
# Get predictions for entire test set
all_predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(all_predictions, axis=1)

# Calculate accuracy correctly
correct = np.sum(predicted_classes == y_test)
manual_accuracy = correct / len(y_test)

print(f"\n✓ FIX #14: Correct manual accuracy calculation")
print(f"Manual accuracy: {manual_accuracy:.4f} ({manual_accuracy*100:.2f}%)")
print(f"Matches model evaluation: {np.isclose(manual_accuracy, test_accuracy)}")

# Show confusion matrix for detailed analysis
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, 
                          target_names=[str(i) for i in range(10)]))

# ==================== SECTION 7: VISUALIZATION ====================

print("\n" + "="*60)
print("VISUALIZING RESULTS")
print("="*60)

# FIX #15: Use correct history keys
# In TensorFlow 2.x, keys are 'accuracy' and 'val_accuracy'
# (not 'acc' and 'val_acc' from TensorFlow 1.x)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_fixed.png', dpi=300, bbox_inches='tight')
print("✓ FIX #15: Using correct history keys ('accuracy', 'val_accuracy')")
print("Training history saved to 'training_history_fixed.png'")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    
    pred_probs = model.predict(X_test[i:i+1], verbose=0)[0]
    pred_class = np.argmax(pred_probs)
    actual_class = y_test[i]
    confidence = pred_probs[pred_class]
    
    color = 'green' if pred_class == actual_class else 'red'
    axes[i].set_title(f'Pred: {pred_class} ({confidence:.2f})\nTrue: {actual_class}',
                     color=color, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
print("Sample predictions saved to 'sample_predictions.png'")

# ==================== SECTION 8: SAVE MODEL ====================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# FIX #16: Use correct file extension
# Recommended formats: .h5 (HDF5) or .keras (native Keras format)
model.save('mnist_model_fixed.h5')
print("✓ FIX #16: Model saved with correct extension (.h5)")
print("Model saved to 'mnist_model_fixed.h5'")

# Also save in Keras native format
model.save('mnist_model_fixed.keras')
print("Model also saved to 'mnist_model_fixed.keras'")

# Save model architecture as JSON
model_json = model.to_json()
with open('mnist_model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
print("Model architecture saved to 'mnist_model_architecture.json'")

# ==================== SECTION 9: FINAL SUMMARY ====================

print("\n" + "="*60)
print("FINAL SUMMARY - ALL FIXES APPLIED")
print("="*60)

print("\n✅ All 16 bugs have been fixed:")
print("  1. Data normalization added")
print("  2. Correct reshape with channel dimension")
print("  3. Proper train/validation split")
print("  4. Appropriate label encoding strategy")
print("  5. Correct input shape in Conv2D")
print("  6. Flatten layer added before Dense")
print("  7. Softmax activation for output")
print("  8. Correct loss function")
print("  9. Variable name typo fixed")
print(" 10. Appropriate learning rate")
print(" 11. Correct validation_data format")
print(" 12. Using correct test data")
print(" 13. Proper prediction interpretation")
print(" 14. Correct accuracy calculation")
print(" 15. Correct history dictionary keys")
print(" 16. Proper model file extension")

print(f"\n📊 Final Results:")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Test Loss: {test_loss:.4f}")
print(f"  • Total Parameters: {model.count_params():,}")
print(f"  • Training Epochs: {len(history.history['accuracy'])}")

print("\n" + "="*60)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*60)
