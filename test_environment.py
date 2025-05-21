#!/usr/bin/env python3
"""
Test script to verify the environment and data loading
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preparation import load_data, preprocess_data, visualize_data, data_distribution
from src.utils import load_and_preprocess_image

def main():
    """Test environment and data loading"""
    # Print versions
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Check the data directory
    data_dir = './data/raw'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    print(f"\nContents of {data_dir}:")
    for item in os.listdir(data_dir):
        print(f"  - {item}")
    
    # Check if we have Train.csv
    train_csv = os.path.join(data_dir, 'Train.csv')
    if not os.path.exists(train_csv):
        print(f"Error: Train.csv does not exist at {train_csv}")
        return
    
    # Try to load a sample image
    try:
        test_csv = os.path.join(data_dir, 'Test.csv')
        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv)
            if not df.empty:
                sample_path = os.path.join(data_dir, df.iloc[0]['Path'])
                print(f"\nTrying to load sample image: {sample_path}")
                if os.path.exists(sample_path):
                    img = cv2.imread(sample_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        print(f"Successfully loaded image with shape: {img.shape}")
                        
                        # Display the image
                        plt.figure(figsize=(6, 6))
                        plt.imshow(img)
                        plt.title(f"Sample Image (Class: {df.iloc[0]['ClassId']})")
                        plt.axis('off')
                        plt.savefig('sample_image.png')
                        print("Sample image saved as 'sample_image.png'")
                    else:
                        print(f"Error: Could not load image using cv2.imread()")
                else:
                    print(f"Error: Sample image file does not exist at {sample_path}")
    except Exception as e:
        print(f"Error testing image loading: {e}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
