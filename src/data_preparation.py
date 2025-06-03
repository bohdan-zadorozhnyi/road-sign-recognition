import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir, csv_file=None):
    """
    Load images and labels from directory structure or CSV file
    
    Args:
        data_dir: Directory containing the images
        csv_file: Optional CSV file with image paths and labels
        
    Returns:
        images: numpy array of images
        labels: numpy array of labels
    """
    images = []
    labels = []
     
    if csv_file:
        # Load data using CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV file with {len(df)} rows")
        
        # Print first few rows for debugging
        print("First few rows of the CSV file:")
        print(df.head())
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing image {idx}/{len(df)}")
                
            img_path = os.path.join(data_dir, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(row['ClassId'])
                else:
                    print(f"Warning: Could not load image at {img_path}")
            else:
                print(f"Warning: Image file does not exist at {img_path}")
    else:
        # Assume directory structure with subfolders for each class
        for class_id in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_id)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(int(class_id))
    
    return np.array(images), np.array(labels)

def preprocess_data(images, target_size=(32, 32)):
    """
    Preprocess images: resize and normalize
    
    Args:
        images: numpy array of images
        target_size: tuple of (width, height) for resizing
        
    Returns:
        preprocessed_images: numpy array of preprocessed images
    """
    preprocessed_images = []
    for img in images:
        img = cv2.resize(img, target_size)
        preprocessed_images.append(img)
    
    preprocessed_images = np.array(preprocessed_images) / 255.0  # Normalize to [0, 1]
    return preprocessed_images

def split_data(images, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        test_size: proportion of data to use for testing
        val_size: proportion of training data to use for validation
        random_state: random seed for reproducibility
        
    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test: split datasets
    """
    # First split: training + validation vs test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: training vs validation
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, 
        test_size=relative_val_size, 
        random_state=random_state,
        stratify=y_train_val
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test


# Custom function to visualize all classes
def visualize_all_classes(images, labels, class_names=None, samples_per_class=1):
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    print(f"Visualizing {n_classes} classes with {samples_per_class} sample(s) per class")
    
    # Calculate grid dimensions
    n_cols = 5  # Fixed number of columns
    n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, n_rows * 3))
    
    for i, class_id in enumerate(unique_classes):
        # Get indices for this class
        indices = np.where(labels == class_id)[0]
        
        # Select random samples
        selected_indices = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        
        # Get class name
        title = f"Class {class_id}"
        if class_names and class_id in class_names:
            title = class_names[class_id]
        
        # Plot the sample
        for j, idx in enumerate(selected_indices):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[idx])
            plt.title(title)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_data(images, labels, class_names=None, samples_per_class=5, figsize=(15, 10)):
    """
    Visualize sample images from the dataset
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        class_names: dictionary mapping class ids to names
        samples_per_class: number of samples to show per class
        figsize: figure size
    """
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    plt.figure(figsize=figsize)
    for i, class_id in enumerate(unique_classes[:n_classes]):
        # Get indices for this class
        indices = np.where(labels == class_id)[0]
        
        # Select random samples
        selected_indices = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        
        for j, idx in enumerate(selected_indices):
            plt.subplot(min(n_classes, 10), samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(images[idx])
            plt.axis('off')
            title = f"Class {class_id}"
            if class_names and class_id in class_names:
                title = class_names[class_id]
            if j == 0:
                plt.title(title)
    
    plt.tight_layout()
    plt.show()

def data_distribution(labels, class_names=None, figsize=(12, 6)):
    """
    Plot the distribution of classes in the dataset
    
    Args:
        labels: numpy array of labels
        class_names: dictionary mapping class ids to names
        figsize: figure size
    """
    # Count instances per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Prepare class names for the plot
    x_labels = []
    for label in unique_labels:
        if class_names and label in class_names:
            x_labels.append(class_names[label])
        else:
            x_labels.append(f"Class {label}")
    
    plt.figure(figsize=figsize)
    sns.barplot(x=unique_labels, y=counts)
    plt.xticks(range(len(x_labels)), x_labels, rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()
