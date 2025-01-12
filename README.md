# Texture Classification Project

This project implements texture classification using two different approaches:
1. Traditional Machine Learning: SVM + LBP Feature Extraction + Zig Zag
2. Deep Learning: CNN with various pre-trained models

## Project Structure

```
project/
│
├── dataset/
│   ├── train/        # Training images
│   └── valid/        # Validation images
│
├── requirements.txt  # Project dependencies
├── training_CNN.py   # CNN training script
└── training_SVM.py   # SVM training script
```

## Setup

1. Create a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create the following directory structure:
     ```
     dataset/
     ├── train/
     │   ├── class1/
     │   ├── class2/
     │   └── ...
     └── valid/
         ├── class1/
         ├── class2/
         └── ...
     ```
   - Place your texture images in their respective class folders

## Training Models

### CNN Model Training

Train the CNN model using different pre-trained architectures:

```bash
python training_CNN.py --pretrain_model [MODEL_NAME] --dataset_name [DATASET] --type_output [OUTPUT_TYPE] --type [DATA_TYPE]
```

Parameters:
- `--pretrain_model`: Choose from 'vgg16', 'resnet50', or 'mobilenet'
- `--dataset_name`: Name of your dataset
- `--type_output`: Type of dataset output ('zigzag' or 'raw')
- `--type`: Type of input data ('raw' or 'zigzag')

Example:
```bash
python training_CNN.py --pretrain_model vgg16 --dataset_name texture_dataset --type_output raw --type raw
```

### SVM Model Training

Train the SVM model with LBP features and Zig Zag scanning:

```bash
python training_SVM.py --pretrain_model [MODEL_NAME] --dataset_name [DATASET] --type_output [OUTPUT_TYPE] --type [DATA_TYPE]
```

Parameters are the same as CNN training.

Example:
```bash
python training_SVM.py --pretrain_model svm_lbp --dataset_name texture_dataset --type_output zigzag --type zigzag
```

## Features

### CNN Approach
- Supports multiple pre-trained models (VGG16, ResNet50, MobileNet)
- Flexible input processing (raw or zigzag)
- Transfer learning capabilities

### SVM Approach
- LBP (Local Binary Pattern) feature extraction
- Zig Zag scanning for feature organization
- Optimized for texture classification

## Notes
- Ensure your texture images are properly organized in the dataset directory
- The project supports both raw image input and zigzag processed features
- Monitor training progress and model performance using provided metrics

## Requirements
See `requirements.txt` for detailed dependencies.

## License
[Your license information here]
