import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import RoadSignRecognitionModel
from utils import load_and_preprocess_image

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Road sign recognition inference')
    
    # Input parameters
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--class_map', type=str, default=None, help='Path to JSON file mapping class IDs to names')
    
    # Image parameters
    parser.add_argument('--input_shape', type=str, default='32,32,3', help='Input shape for the model (height,width,channels)')
    
    return parser.parse_args()

def load_class_map(class_map_path):
    """Load class mapping from JSON file"""
    import json
    
    if not class_map_path or not os.path.exists(class_map_path):
        return None
    
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    
    # Convert keys to integers if they're stored as strings
    return {int(k): v for k, v in class_map.items()}

def main():
    """Main function for inference"""
    # Parse command line arguments
    args = parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Load class mapping
    class_map = load_class_map(args.class_map)
    
    # If class_map is not provided but gtsrb_class_map.json exists, use that
    if class_map is None:
        default_class_map_path = os.path.join(os.path.dirname(__file__), 'gtsrb_class_map.json')
        if os.path.exists(default_class_map_path):
            print(f"Using default class map from {default_class_map_path}")
            class_map = load_class_map(default_class_map_path)
    
    # Load and preprocess image
    print(f"Loading image: {args.image_path}")
    try:
        img = load_and_preprocess_image(args.image_path, target_size=input_shape[:2])
        img_for_display = cv2.imread(args.image_path)
        if img_for_display is None:
            raise ValueError(f"Could not read image: {args.image_path}")
        img_for_display = cv2.cvtColor(img_for_display, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = RoadSignRecognitionModel.load_model(args.model_path)
    
    # Make prediction
    print("Making prediction...")
    input_img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(input_img)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Get class name if available
    class_name = class_map[predicted_class] if class_map and predicted_class in class_map else f"Class {predicted_class}"
    
    print(f"Predicted class: {class_name} (ID: {predicted_class})")
    print(f"Confidence: {confidence:.4f}")
    
    # Display results
    plt.figure(figsize=(6, 6))
    plt.imshow(img_for_display)
    plt.title(f"Prediction: {class_name}\nConfidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
