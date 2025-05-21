# Road Sign Recognition

This project implements a road sign recognition system using deep learning. It can classify different types of road signs from images.

## Project Structure

```
road-sign-recognition/
├── data/
│   ├── raw/         # Raw dataset files
│   └── processed/   # Processed and prepared data
├── src/
│   ├── data_preparation.py  # Functions for data loading and preprocessing
│   ├── model.py             # Model architecture definition
│   ├── train.py             # Script for model training
│   ├── inference.py         # Script for making predictions
│   └── utils.py             # Utility functions
├── models/                  # Saved model files
└── notebooks/               # Jupyter notebooks for exploration
```

## Setup

### Requirements

- Python 3.8+
- TensorFlow 2.8+
- OpenCV
- NumPy, Pandas, Matplotlib, etc.

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/username/road-sign-recognition.git
   cd road-sign-recognition
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Testing the Environment

To verify that your environment is set up correctly and that the data is accessible:

```
source venv/bin/activate
python test_environment.py
```

This script will test:
- Python and library versions
- Data directory structure
- Image loading

## Dataset

This project is designed to work with the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The project expects the dataset to be organized as follows:

```
data/raw/
├── Train.csv       # CSV file with paths and labels for training images
├── Test.csv        # CSV file with paths and labels for test images
├── Meta.csv        # CSV file with metadata about each class
├── Train/          # Directory containing training images
├── Test/           # Directory containing test images
└── Meta/           # Directory containing class prototype images
```

If your dataset is organized differently, you may need to adjust the paths in the scripts or reorganize your data.

### Exploring the Dataset

We provide two Jupyter notebooks for exploring the dataset:

1. `notebooks/data_exploration.ipynb`: Basic exploration of the dataset
2. `notebooks/gtsrb_exploration.ipynb`: More detailed exploration of the GTSRB dataset with our specialized dataset handler

To run these notebooks:

```
source venv/bin/activate
cd notebooks
jupyter notebook
```

## Training

There are two options for training the model:

### Option 1: Using the original training script

```
python src/train.py --data_dir data/raw --csv_file data/raw/Train.csv --output_dir models --epochs 50 --batch_size 64 --augment
```

### Option 2: Using the specialized GTSRB training script (Recommended)

```
python src/train_gtsrb.py --data_dir data/raw --output_dir models --epochs 50 --batch_size 64 --augment --use_roi --visualize
```

Key arguments:
- `--data_dir`: Path to the dataset directory
- `--output_dir`: Directory to save the trained model
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--augment`: Use data augmentation
- `--use_roi`: Use region of interest from CSV files (crop images to focus on signs)
- `--visualize`: Visualize sample images and save diagnostics
- `--limit_samples`: Limit number of samples (for testing, optional)

## Inference

There are two options for making predictions on new images:

### Option 1: Using the original inference script

```
python src/inference.py --image_path path/to/image.jpg --model_path models/final_model.h5 --class_map src/gtsrb_class_map.json
```

### Option 2: Using the specialized GTSRB inference script (Recommended)

```
python src/inference_gtsrb.py --image_path path/to/image.jpg --model_path models/final_model.h5 --use_roi --output_dir results
```

Key arguments:
- `--image_path`: Path to the input image
- `--model_path`: Path to the trained model
- `--class_map`: Optional JSON file mapping class IDs to names (will use default if not provided)
- `--use_roi`: Use region of interest from Test.csv if the image is in the test set
- `--output_dir`: Directory to save output images

## Performance

The model performance may vary depending on the dataset and hyperparameters used. Typical performance metrics on the GTSRB dataset:
- Accuracy: ~95-98%
- Training time: ~30 minutes on a modern GPU

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
