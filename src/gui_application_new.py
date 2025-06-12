import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import traceback
import datetime  # Added for timestamp

# Silence TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf

# Add parent directory to sys.path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from model import RoadSignRecognitionModel, preprocess_image

class RoadSignRecognitionApp:
    def __init__(self, root):
        """Initialize the Road Sign Recognition Application"""
        self.root = root
        self.root.title("Road sign recognition project")
        self.root.geometry("950x550")
        self.root.resizable(False, False)
        
        # Set colors - dark theme
        self.bg_color = "#969696" 
        self.accent_color = "#4E4D51"  
        self.text_color = "#ffffff"  # for text
        self.button_color = "#3c3f44"  # for buttons
        self.root.config(bg=self.bg_color)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Load class mapping
        self.class_map = self.load_class_map()
        
        # Create UI elements
        self.create_ui()
        
    def load_model(self):
        """Load the pre-trained model"""
        # Try multiple potential model paths
        potential_paths = [
            os.path.join(parent_dir, "models", "road_sign_recognition_model.keras"),
            os.path.join(parent_dir, "models", "road_sign_recognition_model.h5"),
            os.path.join(parent_dir, "models", "road_sign_model_checkpoint.keras")
        ]
        
        for model_path in potential_paths:
            if os.path.exists(model_path):
                try:
                    self.model = RoadSignRecognitionModel.load_model(model_path)
                    print(f"Model loaded from {model_path}")
                    return
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    traceback.print_exc()
        
        # If we get here, no model was loaded
        messagebox.showerror("Error", "Could not load model. Please check model paths.")
        print("No model could be loaded. Tried paths:", potential_paths)
    
    def load_class_map(self):
        """Load class mapping from JSON file"""
        # Try multiple potential class map paths
        potential_paths = [
            os.path.join(current_dir, "gtsrb_class_map.json"),
            os.path.join(parent_dir, "src", "gtsrb_class_map.json")
        ]
        
        for class_map_path in potential_paths:
            if os.path.exists(class_map_path):
                try:
                    with open(class_map_path, 'r') as f:
                        class_map = json.load(f)
                    # Convert keys to integers if they're stored as strings
                    return {int(k): v for k, v in class_map.items()}
                except Exception as e:
                    print(f"Error loading class map from {class_map_path}: {e}")
        
        print("No class map could be loaded. Tried paths:", potential_paths)
        return None
    
    def create_ui(self):
        """Create and arrange UI elements"""
        # Create frames
        left_frame = tk.Frame(self.root, width=600, height=550, bg=self.bg_color, bd=1, relief=tk.SUNKEN)
        left_frame.place(x=10, y=10)
        left_frame.pack_propagate(False)
        
        right_frame = tk.Frame(self.root, width=330, height=550, bg=self.bg_color, bd=1, relief=tk.SUNKEN)
        right_frame.place(x=620, y=10)
        right_frame.pack_propagate(False)
        
        # Image display area in left frame with "Load an image to begin" text
        self.image_container = tk.Frame(left_frame, width=580, height=430, bg=self.accent_color)
        self.image_container.pack(side=tk.TOP, pady=(10, 10))
        self.image_container.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_container, text="Load an image to begin", 
                                   bg=self.accent_color, fg=self.text_color, font=("Arial", 12))
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Button container in left frame - at the bottom
        button_frame = tk.Frame(left_frame, bg=self.bg_color)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15)
        
        # Center the button in the frame
        button_center_frame = tk.Frame(button_frame, bg=self.bg_color)
        button_center_frame.pack(side=tk.TOP, anchor=tk.CENTER)
        
        # Load Image button at the bottom of left frame
        load_btn = tk.Button(button_center_frame, text="Load Image", command=self.load_image, 
                            width=15, height=2, relief=tk.RAISED, bg=self.button_color, fg="white",
                            activebackground="#2980b9", activeforeground="white", font=("Arial", 10))
        load_btn.pack(padx=5)
        load_btn.pack(pady=20)
        
        # Right frame components
        # Process section title with horizontal line below it
        process_frame = tk.Frame(right_frame, bg=self.bg_color)
        process_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
        
        process_label = tk.Label(process_frame, text="Process", bg=self.bg_color, fg=self.text_color,
                               font=("Arial", 12, "bold"))
        process_label.pack(side=tk.LEFT, anchor=tk.W)
        
        separator_frame = tk.Frame(right_frame, height=2, bg="#7f8c8d")
        separator_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        
        # Process button on the right side
        process_button_frame = tk.Frame(right_frame, bg=self.bg_color)
        process_button_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.process_btn = tk.Button(process_button_frame, text="Process", command=self.process_image, 
                                  width=15, height=2, relief=tk.RAISED, bg=self.button_color, fg="white", 
                                  activebackground="#2980b9", activeforeground="white", font=("Arial", 10))
        self.process_btn.pack(side=tk.RIGHT, pady=10)
        self.process_btn.config(state=tk.DISABLED)
        
        # Results section title with horizontal line below it
        results_frame = tk.Frame(right_frame, bg=self.bg_color)
        results_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(20, 0))
        
        results_label = tk.Label(results_frame, text="Results", bg=self.bg_color, fg=self.text_color,
                               font=("Arial", 12, "bold"))
        results_label.pack(side=tk.LEFT, anchor=tk.W)
        
        separator_frame2 = tk.Frame(right_frame, height=2, bg="#7f8c8d")
        separator_frame2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        
        # Results display area
        self.result_text = tk.Text(right_frame, width=35, height=15, wrap=tk.WORD, bd=1, relief=tk.SUNKEN, 
                                bg=self.accent_color, fg=self.text_color, insertbackground=self.text_color)
        self.result_text.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        
        # Save button frame at the bottom right
        save_button_frame = tk.Frame(right_frame, bg=self.bg_color)
        save_button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=15)
        
        # Save button (initially disabled) placed at the bottom of right frame
        self.save_btn = tk.Button(save_button_frame, text="Save Results", command=self.save_results, 
                               width=15, height=2, relief=tk.RAISED, bg=self.button_color, fg="white",
                               activebackground="#2980b9", activeforeground="white", font=("Arial", 10))
        self.save_btn.pack(side=tk.RIGHT)
        self.save_btn.config(state=tk.DISABLED)
        self.save_btn.pack(pady=25)
        
        # Footer
        footer_text = "Road Sign Recognition Application v1.0 | © 2025 | Developed by Andrei Marshyn, Anastasiia Kuvshynova, Bohdan Zadorozhnyi, Yurii Demoshenko                          "
        footer_label = tk.Label(self.root, text=footer_text, bg="#2a2727", fg="#ffffff", font=("Arial", 10))
        footer_label.place(x=0, y=528)
    
    def load_image(self):
        """Load an image file and display it"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        # Store the file path for processing
        self.current_image_path = file_path
        self.file_name = os.path.basename(file_path)
        
        # Load and display the image
        try:
            # Use OpenCV to load the image (for better compatibility with the model)
            cv_img = cv2.imread(file_path)
            if cv_img is None:
                raise ValueError("Could not load image file")
            
            # Convert BGR to RGB for display
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(cv_img)
            
            # Resize for display
            pil_img = self.resize_image(pil_img, (580, 430))  # Updated to match new container size
            
            # Convert to Tkinter PhotoImage
            self.display_img = ImageTk.PhotoImage(pil_img)
            
            # Update image display
            self.image_label.config(image=self.display_img, text="")
            
            # Enable process button
            self.process_btn.config(state=tk.NORMAL)
            
            # Clear previous results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error loading image: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.image_label.config(text="Error loading image", image="")
    
    def resize_image(self, img, size):
        """Resize image while maintaining aspect ratio"""
        original_width, original_height = img.size
        target_width, target_height = size
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        # Determine new dimensions
        if original_ratio > target_ratio:
            # Image is wider than target area
            new_width = target_width
            new_height = int(new_width / original_ratio)
        else:
            # Image is taller than target area
            new_height = target_height
            new_width = int(new_height * original_ratio)
        
        # Resize image
        try:
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        except AttributeError:
            # LANCZOS might not be available in older PIL versions
            resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Create a blank image with the target size and background matching the UI theme
        new_img = Image.new("RGB", size, color="#34495e")  # Match the accent color
        
        # Calculate position to paste the resized image (centered)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img
    
    def process_image(self):
        """Process the loaded image and display the result"""
        if not hasattr(self, 'current_image_path'):
            messagebox.showinfo("Info", "Please load an image first")
            return
            
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded properly")
            return
        
        try:
            # Show processing indicator
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Processing image...\nPlease wait...")
            self.result_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Using a safe preprocessing function that handles tensorflow deprecation warnings
            # and potential issues with the image loading
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError("Failed to load image for processing")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))
            img = img / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Make prediction
            try:
                prediction = self.model.predict(img_array)
                predicted_class_id = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_id])  # Convert to Python float
            except Exception as pred_error:
                raise RuntimeError(f"Error during model prediction: {pred_error}")
            
            # Get class name
            if self.class_map and predicted_class_id in self.class_map:
                class_name = self.class_map[predicted_class_id]
            else:
                class_name = f"Class {predicted_class_id}"
            
            # Store prediction results for saving
            self.last_prediction = {
                'class_id': int(predicted_class_id),
                'class_name': class_name,
                'confidence': float(confidence),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Display result
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            
            result_str = f"Result for '{self.file_name}':\n\n{class_name}\n\nConfidence: {confidence:.4f}"
            self.result_text.insert(tk.END, result_str)
            self.result_text.config(state=tk.DISABLED)
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            
            print(f"Processed {self.file_name}: {class_name} (Confidence: {confidence:.4f})")
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, error_msg)
            self.result_text.config(state=tk.DISABLED)
            
            # Disable save button
            self.save_btn.config(state=tk.DISABLED)
            
            # Show error dialog
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
    
    def save_results(self):
        """Save the prediction results to a text file"""
        if not hasattr(self, 'current_image_path') or not hasattr(self, 'last_prediction'):
            messagebox.showinfo("Info", "No results to save. Please process an image first.")
            return
        
        # Get a file path to save to
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"prediction_{os.path.splitext(self.file_name)[0]}.txt"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                # Write prediction results
                f.write(f"Road Sign Recognition Results\n")
                f.write(f"==========================\n\n")
                f.write(f"Image: {self.file_name}\n")
                f.write(f"Prediction: {self.last_prediction['class_name']}\n")
                f.write(f"Confidence: {self.last_prediction['confidence']:.4f}\n")
                f.write(f"Class ID: {self.last_prediction['class_id']}\n\n")
                f.write(f"Generated on: {self.last_prediction['timestamp']}\n")
                
            messagebox.showinfo("Success", f"Results saved to {file_path}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    root = tk.Tk()
    app = RoadSignRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
