#!/usr/bin/env python3
"""
GTSRB Inference Script for making predictions with a trained model
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from model import RoadSignRecognitionModel
from utils import load_and_preprocess_image

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Road sign recognition inference')
    
    # Input parameters
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.h5 file)')
    parser.add_argument('--class_map', type=str, default=None, help='Path to JSON file mapping class IDs to names')
    
    # Image parameters
    parser.add_argument('--input_shape', type=str, default='32,32,3', help='Input shape for the model (height,width,channels)')
    parser.add_argument('--use_roi', action='store_true', help='Crop to ROI if image is from the test set')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output images')
    
    return parser.parse_args()

def load_class_map(class_map_path):
    """Load class mapping from JSON file"""
    if not class_map_path or not os.path.exists(class_map_path):
        # Try default locations
        default_locations = [
            os.path.join(os.path.dirname(__file__), 'gtsrb_class_map.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'gtsrb_class_map.json')
        ]
        
        # Also check if there's a class_map.json in the same directory as the model
        if args.model_path:
            model_dir = os.path.dirname(args.model_path)
            default_locations.append(os.path.join(model_dir, 'class_map.json'))
        
        for location in default_locations:
            if os.path.exists(location):
                class_map_path = location
                print(f"Using default class map from {location}")
                break
    
    if class_map_path and os.path.exists(class_map_path):
        try:
            with open(class_map_path, 'r') as f:
                class_map = json.load(f)
            
            # Convert keys to integers if they're stored as strings
            return {int(k): v for k, v in class_map.items()}
        except Exception as e:
            print(f"Error loading class map: {e}")
    
    return None

def get_image_roi(image_path, test_csv_path=None):
    """
    Get the region of interest for an image if it's in the test set
    
    Args:
        image_path: Path to the image
        test_csv_path: Path to the Test.csv file
        
    Returns:
        roi: Tuple of (x1, y1, x2, y2) or None if not found
    """
    if not test_csv_path:
        # Try to find Test.csv in standard locations
        possible_locations = [
            os.path.join(os.path.dirname(os.path.dirname(image_path)), 'Test.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(image_path))), 'Test.csv'),
            './data/raw/Test.csv',
            '../data/raw/Test.csv'
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                test_csv_path = loc
                break
    
    if not test_csv_path or not os.path.exists(test_csv_path):
        return None
    
    # Get just the filename without path
    image_filename = os.path.basename(image_path)
    
    # Load the CSV
    import pandas as pd
    try:
        df = pd.read_csv(test_csv_path)
        
        # Find the image in the CSV
        for _, row in df.iterrows():
            if image_filename in row['Path']:
                # Found the image, extract ROI
                if all(col in row for col in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']):
                    return (
                        int(row['Roi.X1']), 
                        int(row['Roi.Y1']), 
                        int(row['Roi.X2']), 
                        int(row['Roi.Y2'])
                    )
    except Exception as e:
        print(f"Error getting ROI from CSV: {e}")
    
    return None

def crop_to_roi(image, roi):
    """Crop an image to a region of interest"""
    if roi is None or image is None:
        return image
    
    x1, y1, x2, y2 = roi
    if x1 < x2 and y1 < y2 and x2 <= image.shape[1] and y2 <= image.shape[0]:
        return image[y1:y2, x1:x2]
    
    return image

def main():
    """Main function for inference"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Load class mapping
    class_map = load_class_map(args.class_map)
    
    # Load image
    print(f"Loading image: {args.image_path}")
    try:
        # Read the image with OpenCV
        img = cv2.imread(args.image_path)
        if img is None:
            raise ValueError(f"Could not read image: {args.image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save a copy of the original image
        original_img = img.copy()
        
        # Crop to ROI if requested
        if args.use_roi:
            roi = get_image_roi(args.image_path)
            if roi:
                print(f"Cropping to ROI: {roi}")
                img = crop_to_roi(img, roi)
        
        # Resize and normalize
        img_resized = cv2.resize(img, input_shape[:2])
        img_normalized = img_resized / 255.0
        
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return
    
    # Load model
    print(f"Loading model: {args.model_path}")
    try:
        model = RoadSignRecognitionModel.load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    print("Making prediction...")
    input_img = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    prediction = model.predict(input_img)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Get class name if available
    if class_map and predicted_class in class_map:
        class_name = class_map[predicted_class]
    else:
        class_name = f"Class {predicted_class}"
    
    print(f"Predicted class: {class_name} (ID: {predicted_class})")
    print(f"Confidence: {confidence:.4f}")
    
    # Display and save results
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Preprocessed image with prediction
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(f"Prediction: {class_name}\nConfidence: {confidence:.4f}")
    plt.axis('off')
    
    # Save figure
    output_path = os.path.join(args.output_dir, 'prediction_result.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")
    
    # Display top 5 predictions if we have more than 5 classes
    if prediction.shape[1] >= 5:
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        top_probabilities = prediction[0][top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(5), top_probabilities)
        plt.yticks(range(5), [class_map.get(idx, f"Class {idx}") for idx in top_indices])
        plt.xlabel('Probability')
        plt.title('Top 5 Predictions')
        plt.tight_layout()
        
        # Save top 5 predictions
        top5_path = os.path.join(args.output_dir, 'top5_predictions.png')
        plt.savefig(top5_path)
        print(f"Top 5 predictions saved to {top5_path}")

if __name__ == "__main__":
    args = parse_args()
    main()
