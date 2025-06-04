import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# --- Load model once ---
def load_model():
    try:
        model = tf.keras.models.load_model("models\\road_sign_recognition_model.h5")  # <<< replace with your model path
        print("Model loaded successfully.")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return None

# --- Predict image ---
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((32, 32), Image.Resampling.LANCZOS)  # Adjust to your model input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = np.max(preds)

        class_names = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
            'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
            'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at intersection',
            'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
            'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
            'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
            'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
            'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
            'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
        ]

        result_str = f"{class_names[class_idx]} (Confidence: {confidence:.2f})"
        return result_str

    except Exception as e:
        return f"Error processing image: {e}"

# --- Create main window ---
root = tk.Tk()
root.title("Road sign recognition project")
root.geometry("900x650")
root.resizable(False, False)

# --- Load model ---
model = load_model()

# --- Functions ---
def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            image_label.config(image=img_tk, text="")
            image_label.image = img_tk

            result_label.config(text=f"Result for '{file_path.split('/')[-1]}':\n")
            result_text.set("")

            root.current_image_path = file_path

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

def process_image():
    if hasattr(root, 'current_image_path'):
        result = predict_image(root.current_image_path)
        result_text.set(result)
    else:
        messagebox.showwarning("No Image", "Please load an image first.")

# --- Layout ---
left_frame = tk.Frame(root, width=600, height=600, bd=2, relief=tk.SUNKEN)
left_frame.pack(side=tk.LEFT, padx=5, pady=5)

right_frame = tk.Frame(root, width=300, height=600, bd=2, relief=tk.SUNKEN)
right_frame.pack(side=tk.RIGHT, padx=5, pady=5)

# Left frame content
image_label = tk.Label(left_frame, text="Load an image to begin", font=("Arial", 12), fg="gray")
image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

load_button = tk.Button(left_frame, text="Load Image", command=load_image)
load_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

# Right frame content
process_button = tk.Button(right_frame, text="Process", width=20, height=1, command=process_image)
process_button.pack(pady=20)

result_label = tk.Label(right_frame, text="Result for '':", font=("Arial", 11))
result_label.pack(anchor='w', padx=10, pady=(20, 5))

result_text = tk.StringVar()
result_output = tk.Label(right_frame, textvariable=result_text, font=("Arial", 12))
result_output.pack(anchor='w', padx=10)

# Footer / Status bar
footer = tk.Label(root, text="Road Sign Recognition Application v1.0 | © 2025 | Developed by Andrei Marshyn, Anastasiia Kuvshynova, Bohdan Zadorozhnyi, Yurii Demoshenko", bd=1, relief=tk.SUNKEN, anchor=tk.W)
footer.pack(side=tk.BOTTOM, fill=tk.X)

# Run the app
root.mainloop()