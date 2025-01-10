# Tuberculosis X-ray Segmentation Model

## Overview
This project implements a complete pipeline for segmenting tuberculosis in X-ray images using the SegFormer model from Hugging Face Transformers. The model is designed to identify and segment tuberculosis regions in chest X-ray images.

## Features
- Custom dataset handling for X-ray images and masks
- Automated image resizing and padding
- Data augmentation pipeline
- Training and validation split functionality
- Model training with checkpoint saving
- Visualization tools for predictions

## Requirements
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- PIL (Python Imaging Library)
- pandas
- numpy
- scikit-learn
- tqdm

## Project Structure
.
├── datasets/
│ ├── image/ # X-ray images
│ ├── mask/ # Segmentation masks
│ └── MetaData.csv # Dataset metadata
├── TB_Segmentation.ipynb
└── best_model.pth # Saved model weights (after training)


## Usage
1. Prepare your dataset:
   - Place X-ray images in `datasets/image/`
   - Place corresponding masks in `datasets/mask/`
   - Update metadata file in `datasets/MetaData.csv`

2. Run the training pipeline:
   - Open `TB_Segmentation.ipynb`
   - Execute all cells to train the model
   - Best model weights will be saved as 'best_model.pth'

## Model Details
- Architecture: SegFormer (nvidia/mit-b0)
- Input size: 128x128 pixels
- Output: Binary segmentation mask (tuberculosis vs background)
- Training parameters:
  - Batch size: 8
  - Optimizer: AdamW
  - Learning rate: 2e-5

## Visualization
The project includes a visualization function `visualize_prediction()` that displays:
- Original X-ray image
- Predicted segmentation mask

## License
[Add your license information here]

## Contact
[Add your contact information here]