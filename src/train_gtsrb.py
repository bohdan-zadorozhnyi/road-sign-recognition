#!/usr/bin/env python3
"""
GTSRB Training Script using the GTSRBDataset class
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Import our modules
from gtsrb_dataset import GTSRBDataset
from model import RoadSignRecognitionModel
from utils import confusion_matrix_plot, save_model_history, create_tf_dataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a road sign recognition model on GTSRB dataset')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Directory containing the GTSRB dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save the trained model')
    parser.add_argument('--use_roi', action='store_true', help='Use region of interest from CSV file')
    
    # Model parameters
    parser.add_argument('--input_shape', type=str, default='32,32,3', help='Input shape for the model (height,width,channels)')
    parser.add_argument('--num_classes', type=int, default=43, help='Number of classes in the dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples (for testing)')
    
    # Validation and testing
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of training data to use for validation')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample images')
    
    return parser.parse_args()

def main():
    """Main function to train the model"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = GTSRBDataset(args.data_dir, use_roi=args.use_roi)
    
    # Load training data
    print("Loading training data...")
    train_images, train_labels = dataset.load_train_data(limit=args.limit_samples)
    print(f"Loaded {len(train_images)} images with {len(np.unique(train_labels))} classes")
    
    # Visualize class distribution
    if args.visualize:
        print("Visualizing class distribution...")
        dataset.plot_class_distribution(train_labels)
    
    # Visualize sample images
    if args.visualize:
        print("Visualizing sample images...")
        dataset.visualize_samples(train_images, train_labels)
    
    # Preprocess images
    print("Preprocessing images...")
    processed_images = dataset.preprocess_data(train_images, target_size=input_shape[:2])
    
    # Split data into train, validation, and test sets
    print("Splitting data...")
    x_train, x_val, x_test, y_train, y_val, y_test = dataset.split_data(
        processed_images, train_labels, test_size=args.test_size, val_size=args.val_size, random_state=args.random_seed
    )
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Create TensorFlow datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_tf_dataset(x_train, y_train, batch_size=args.batch_size, augment=args.augment)
    val_dataset = create_tf_dataset(x_val, y_val, batch_size=args.batch_size, shuffle=False)
    
    # Save some sample images to verify data loading
    if args.visualize:
        print("Saving sample training images...")
        plt.figure(figsize=(10, 5))
        for i in range(min(10, len(x_train))):
            plt.subplot(2, 5, i+1)
            plt.imshow(x_train[i])
            plt.title(f"Class {y_train[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sample_training_images.png'))
    
    # Number of classes based on actual data
    num_classes = len(np.unique(train_labels))
    if args.num_classes != num_classes:
        print(f"Warning: Specified {args.num_classes} classes, but dataset has {num_classes} classes.")
        print(f"Using {num_classes} classes based on data.")
    
    # Create and compile model
    print("Creating model...")
    model = RoadSignRecognitionModel(input_shape=input_shape, num_classes=num_classes)
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.train(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Save training history
    save_model_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate on test set
    print("Evaluating model...")
    test_dataset = create_tf_dataset(x_test, y_test, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
    
    # Get predictions for confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    confusion_matrix_plot(y_test, y_pred, class_names=dataset.class_names)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save model class mapping
    import json
    with open(os.path.join(args.output_dir, 'class_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in dataset.class_names.items()}, f, indent=2)
    
    # Save the final model
    model.save_model(os.path.join(args.output_dir, 'final_model.h5'))
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model.h5')}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
