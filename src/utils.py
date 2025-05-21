import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    
    return img


def apply_augmentation(img, augmentation_type='random'):
    """
    Apply image augmentation
    
    Args:
        img: Input image as numpy array
        augmentation_type: Type of augmentation to apply
        
    Returns:
        Augmented image
    """
    if augmentation_type == 'random':
        # Apply random combination of augmentations
        augmentations = ['rotate', 'flip', 'brightness', 'shift']
        chosen = np.random.choice(augmentations)
        return apply_augmentation(img, chosen)
    
    elif augmentation_type == 'rotate':
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))
    
    elif augmentation_type == 'flip':
        # Horizontal flip
        return cv2.flip(img, 1)
    
    elif augmentation_type == 'brightness':
        # Brightness adjustment
        # Convert to uint8 if it's float
        if img.dtype != np.uint8:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img.copy()
            
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        value = np.random.uniform(0.7, 1.3)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * value, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Return in the same format as input
        if img.dtype != np.uint8:
            return result.astype(np.float32) / 255.0
        else:
            return result
    
    elif augmentation_type == 'shift':
        # Random shift
        h, w = img.shape[:2]
        tx = np.random.uniform(-w*0.1, w*0.1)
        ty = np.random.uniform(-h*0.1, h*0.1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h))
    
    else:
        return img


def class_accuracy(y_true, y_pred):
    """
    Calculate accuracy for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (as integers, not one-hot encoded)
        
    Returns:
        Dictionary mapping class IDs to their accuracy
    """
    classes = np.unique(y_true)
    accuracy = {}
    
    for cls in classes:
        # Get indices for this class
        indices = np.where(y_true == cls)[0]
        
        # Get predictions and true values for this class
        class_pred = y_pred[indices]
        class_true = y_true[indices]
        
        # Calculate accuracy
        accuracy[cls] = np.mean(class_pred == class_true)
    
    return accuracy


def confusion_matrix_plot(y_true, y_pred, class_names=None, figsize=(12, 10)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Dictionary mapping class IDs to names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class labels for the plot
    classes = np.unique(y_true)
    if class_names:
        labels = [class_names.get(cls, f"Class {cls}") for cls in classes]
    else:
        labels = [f"Class {cls}" for cls in classes]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def save_model_history(history, filepath):
    """
    Save training history plot
    
    Args:
        history: Keras history object from model.fit
        filepath: Path to save the plot
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 4))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    
def create_tf_dataset(images, labels, batch_size=32, shuffle=True, augment=False):
    """
    Create a TensorFlow dataset for training or evaluation
    
    Args:
        images: Numpy array of images
        labels: Numpy array of labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        augment: Whether to apply data augmentation
        
    Returns:
        TensorFlow dataset
    """
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Apply shuffling if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    # Apply augmentation if requested
    if augment:
        def augmentation_fn(image, label):
            # Convert to numpy, apply augmentation, convert back to tensor
            img_np = image.numpy()
            
            # Convert to uint8 format if necessary for OpenCV functions
            if img_np.dtype != np.uint8 and np.max(img_np) <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented_img = apply_augmentation(img_np)
            
            # Convert back to float32 [0, 1] range for TensorFlow
            if np.max(augmented_img) > 1.0:
                augmented_img = augmented_img.astype(np.float32) / 255.0
                
            return tf.convert_to_tensor(augmented_img, dtype=tf.float32), label
        
        dataset = dataset.map(lambda x, y: tf.py_function(
            augmentation_fn, [x, y], [tf.float32, tf.int64]
        ))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
