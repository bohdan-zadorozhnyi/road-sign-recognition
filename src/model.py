import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

class RoadSignRecognitionModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=43, dropout_rate=0.5):
        """
        Initialize the Road Sign Recognition model
        Args:
            input_shape: Shape of the input images
            num_classes: Number of road sign classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build a CNN model for road sign classification
        Returns:
            A compiled Keras model
        """
        model = models.Sequential()
        
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Train the model on the provided data
        """
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
            
        return self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data
        """
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, images):
        """
        Make predictions on new images
        """
        return self.model.predict(images)
    
    def save_model(self, path):
        """
        Save the model to the specified path
        """
        self.model.save(path)
    
    @classmethod
    def load_model(cls, path):
        """
        Load a pre-trained model from the specified path
        """
        model_instance = cls()
        model_instance.model = models.load_model(path)
        return model_instance

def preprocess_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess an image for prediction
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension
