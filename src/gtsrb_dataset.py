#!/usr/bin/env python3
"""
GTSRB Dataset handler to load and process the German Traffic Sign Recognition Benchmark dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

class GTSRBDataset:
    """Handler for the German Traffic Sign Recognition Benchmark dataset"""
    
    def __init__(self, data_dir, use_roi=True):
        """
        Initialize the GTSRB dataset
        
        Args:
            data_dir: Directory containing the raw dataset (with Train.csv, Test.csv)
            use_roi: Whether to crop images to their region of interest
        """
        self.data_dir = data_dir
        self.train_csv = os.path.join(data_dir, 'Train.csv')
        self.test_csv = os.path.join(data_dir, 'Test.csv')
        self.meta_csv = os.path.join(data_dir, 'Meta.csv')
        self.class_map_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          'src', 'gtsrb_class_map.json')
        self.use_roi = use_roi
        
        # Verify files exist
        self._verify_files()
        
        # Load class names
        self.class_names = self._load_class_names()
    
    def _verify_files(self):
        """Verify that required files exist"""
        if not os.path.exists(self.train_csv):
            raise FileNotFoundError(f"Train.csv not found at {self.train_csv}")
        if not os.path.exists(self.test_csv):
            raise FileNotFoundError(f"Test.csv not found at {self.test_csv}")
    
    def _load_class_names(self):
        """Load class names from JSON file or create from Meta.csv"""
        class_names = {}
        
        # Try to load from JSON file first
        if os.path.exists(self.class_map_file):
            try:
                with open(self.class_map_file, 'r') as f:
                    class_names = json.load(f)
                # Convert string keys to integers
                class_names = {int(k): v for k, v in class_names.items()}
                print(f"Loaded {len(class_names)} class names from {self.class_map_file}")
                return class_names
            except Exception as e:
                print(f"Error loading class map from JSON: {e}")
        
        # If that fails and Meta.csv exists, create class names from there
        if os.path.exists(self.meta_csv):
            try:
                meta_df = pd.read_csv(self.meta_csv)
                for _, row in meta_df.iterrows():
                    class_id = row['ClassId']
                    # Use Path as the name (removing Meta/ and .png)
                    name = row['Path'].replace('Meta/', '').replace('.png', '')
                    class_names[class_id] = f"Sign {name}"
                
                print(f"Created {len(class_names)} class names from Meta.csv")
                return class_names
            except Exception as e:
                print(f"Error creating class names from Meta.csv: {e}")
        
        # If all else fails, create generic class names
        unique_classes = set()
        try:
            train_df = pd.read_csv(self.train_csv)
            unique_classes.update(train_df['ClassId'].unique())
            
            test_df = pd.read_csv(self.test_csv)
            unique_classes.update(test_df['ClassId'].unique())
            
            for class_id in unique_classes:
                class_names[class_id] = f"Class {class_id}"
            
            print(f"Created {len(class_names)} generic class names")
        except Exception as e:
            print(f"Error creating generic class names: {e}")
        
        return class_names
    
    def load_train_data(self, limit=None):
        """
        Load training data
        
        Args:
            limit: Optional limit on number of samples to load (for testing)
            
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        return self._load_data_from_csv(self.train_csv, limit)
    
    def load_test_data(self, limit=None):
        """
        Load test data
        
        Args:
            limit: Optional limit on number of samples to load (for testing)
            
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        return self._load_data_from_csv(self.test_csv, limit)
    
    def _load_data_from_csv(self, csv_file, limit=None):
        """
        Load data from a CSV file
        
        Args:
            csv_file: Path to the CSV file
            limit: Optional limit on number of samples to load
            
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        images = []
        labels = []
        
        # Load data using CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV file with {len(df)} rows")
        
        # Apply limit if specified
        if limit is not None:
            df = df.head(limit)
            print(f"Limited to first {limit} samples")
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"Processing image {idx}/{len(df)}")
                
            img_path = os.path.join(self.data_dir, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Crop to ROI if requested and available
                    if self.use_roi and all(col in row for col in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']):
                        try:
                            x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
                            if x1 < x2 and y1 < y2:  # Valid ROI
                                img = img[y1:y2, x1:x2]
                        except Exception as e:
                            print(f"Error cropping ROI for image {img_path}: {e}")
                    
                    images.append(img)
                    labels.append(row['ClassId'])
                else:
                    print(f"Warning: Could not load image at {img_path}")
            else:
                print(f"Warning: Image file does not exist at {img_path}")
        
        print(f"Loaded {len(images)} images with {len(set(labels))} unique classes")
        # Return images as a list to handle different sizes, but convert labels to numpy array
        return images, np.array(labels)
    
    def preprocess_data(self, images, target_size=(32, 32)):
        """
        Preprocess images: resize and normalize
        
        Args:
            images: list or numpy array of images (handles variable sizes)
            target_size: tuple of (width, height) for resizing
            
        Returns:
            preprocessed_images: numpy array of preprocessed images (uniform size)
        """
        preprocessed_images = []
        for i, img in enumerate(images):
            if i % 1000 == 0 and i > 0:
                print(f"Preprocessed {i}/{len(images)} images")
            
            try:
                # Handle potentially different sized images
                img = cv2.resize(img, target_size)
                preprocessed_images.append(img)
            except Exception as e:
                print(f"Error preprocessing image {i}: {e}")
                # Add a blank placeholder image instead of failing
                blank_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                preprocessed_images.append(blank_img)
        
        preprocessed_images = np.array(preprocessed_images) / 255.0  # Normalize to [0, 1]
        return preprocessed_images
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.2, random_state=42):
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
    
    def split_train_val(self, images, labels, val_size=0.2, random_state=42):
        """
        Split data into training and validation sets only (when a separate test set exists)
        
        Args:
            images: numpy array of images
            labels: numpy array of labels
            val_size: proportion of data to use for validation
            random_state: random seed for reproducibility
            
        Returns:
            x_train, x_val, y_train, y_val: split datasets
        """
        # Split directly into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            images, labels, 
            test_size=val_size, 
            random_state=random_state,
            stratify=labels
        )
        
        print(f"Training set: {len(x_train)} samples ({100 * (1 - val_size):.0f}%)")
        print(f"Validation set: {len(x_val)} samples ({100 * val_size:.0f}%)")
        
        return x_train, x_val, y_train, y_val
    
    def visualize_samples(self, images, labels, samples_per_class=2, max_classes=10, figsize=(15, 10)):
        """
        Visualize sample images from the dataset
        
        Args:
            images: list or numpy array of images
            labels: numpy array of labels
            samples_per_class: number of samples to show per class
            max_classes: maximum number of classes to show
            figsize: figure size
        """
        # Get unique classes
        unique_classes = np.unique(labels)
        n_classes = min(len(unique_classes), max_classes)
        
        plt.figure(figsize=figsize)
        for i, class_id in enumerate(unique_classes[:n_classes]):
            # Get indices for this class
            indices = np.where(labels == class_id)[0]
            
            # Select samples
            selected_indices = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
            
            # Get class name
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            
            for j, idx in enumerate(selected_indices):
                try:
                    plt.subplot(n_classes, samples_per_class, i * samples_per_class + j + 1)
                    plt.imshow(images[idx])
                    plt.axis('off')
                    if j == 0:
                        plt.title(f"{class_name}")
                except Exception as e:
                    print(f"Error displaying image at index {idx}: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, labels, figsize=(15, 6)):
        """
        Plot the distribution of classes in the dataset
        
        Args:
            labels: numpy array of labels
            figsize: figure size
        """
        # Count instances per class
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by class ID
        sorted_indices = np.argsort(unique_labels)
        unique_labels = unique_labels[sorted_indices]
        counts = counts[sorted_indices]
        
        # Get class names for the plot
        x_labels = [self.class_names.get(label, f"Class {label}") for label in unique_labels]
        
        plt.figure(figsize=figsize)
        plt.bar(range(len(unique_labels)), counts)
        plt.xticks(range(len(unique_labels)), x_labels, rotation=90)
        plt.xlabel('Class')
        plt.ylabel('Number of samples')
        plt.title('Class Distribution')
        plt.tight_layout()
        plt.show()
        
        # Print stats
        print(f"Total number of samples: {sum(counts)}")
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Average samples per class: {sum(counts)/len(unique_labels):.1f}")
        print(f"Min samples for a class: {min(counts)} (Class {unique_labels[np.argmin(counts)]})")
        print(f"Max samples for a class: {max(counts)} (Class {unique_labels[np.argmax(counts)]})")


if __name__ == "__main__":
    # Simple test of the class
    data_dir = './data/raw'
    dataset = GTSRBDataset(data_dir)
    
    # Load a small sample of training data for testing
    train_images, train_labels = dataset.load_train_data(limit=100)
    
    # Preprocess
    train_images_processed = dataset.preprocess_data(train_images)
    
    # Visualize
    dataset.visualize_samples(train_images, train_labels)
    
    # Class distribution
    dataset.plot_class_distribution(train_labels)
