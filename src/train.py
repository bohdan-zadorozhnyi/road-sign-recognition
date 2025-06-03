import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import wandb

# Import our modules
from data_preparation import load_data, preprocess_data, split_data, visualize_data, data_distribution
from model import RoadSignRecognitionModel
from utils import confusion_matrix_plot, save_model_history, create_tf_dataset

class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({
            "train_loss": logs.get("loss"),
            "train_accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy")
        })

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a road sign recognition model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--csv_file', type=str, default=None, help='Optional CSV file with image paths and labels')
    parser.add_argument('--output_dir', type=str, default='../models', help='Directory to save the trained model')
    
    # Model parameters
    parser.add_argument('--input_shape', type=str, default='32,32,3', help='Input shape for the model (height,width,channels)')
    parser.add_argument('--num_classes', type=int, default=43, help='Number of classes in the dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
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
    
    # Load data
    print("Loading data...")
    
    # Default to using Train.csv if no csv_file is provided
    if args.csv_file is None and os.path.exists(os.path.join(args.data_dir, 'Train.csv')):
        args.csv_file = os.path.join(args.data_dir, 'Train.csv')
        print(f"Using default Train.csv file: {args.csv_file}")
        
    images, labels = load_data(args.data_dir, args.csv_file)
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} classes")
    
    # Data distribution
    data_distribution(labels)
    
    # Visualize sample images if requested
    if args.visualize:
        print("Visualizing sample images...")
        visualize_data(images, labels)
    
    # Preprocess images
    print("Preprocessing images...")
    images = preprocess_data(images, target_size=input_shape[:2])
    
    # Split data
    print("Splitting data...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        images, labels, test_size=args.test_size, val_size=args.val_size, random_state=args.random_seed
    )
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(x_train, y_train, batch_size=args.batch_size, augment=args.augment)
    val_dataset = create_tf_dataset(x_val, y_val, batch_size=args.batch_size, shuffle=False)
    
    # Create and compile model
    print("Creating model...")
    model = RoadSignRecognitionModel(input_shape=input_shape, num_classes=args.num_classes)


    
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
        ),
        # CustomWandbCallback()
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
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions and confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    confusion_matrix_plot(y_test, y_pred)
    
    # Save the final model
    model.save_model(os.path.join(args.output_dir, 'final_model.h5'))
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model.h5')}")

if __name__ == "__main__":
    main()
