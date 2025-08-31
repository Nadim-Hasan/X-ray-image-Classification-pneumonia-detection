hest X-ray Pneumonia Detection
A deep learning project for classifying chest X-ray images into Normal and Pneumonia categories using convolutional neural networks.

Project Overview
This project uses a Kaggle dataset of chest X-ray images to build a classification model that can detect pneumonia from X-ray scans. The dataset contains 5,856 validated X-ray images categorized into two classes: Normal and Pneumonia.

Dataset
The dataset is sourced from Kaggle:

Dataset Name: Chest X-Ray Images (Pneumonia)

Source: Kaggle Dataset

Total Images: 5,856 JPEG images

Classes:

Normal: 1,583 images

Pneumonia: 4,273 images

The dataset is divided into three folders:

train

test

val

Setup Instructions
Prerequisites
Python 3.6+

Kaggle account and API credentials


Project Structure:
├── data/
│   ├── chest_xray/
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── test/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   └── val/
│   │       ├── NORMAL/
│   │       └── PNEUMONIA/
├── models/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── requirements.txt
└── README.md



Model Architecture
The project uses a Convolutional Neural Network (CNN) with:

Multiple convolutional layers with ReLU activation

Max pooling layers

Dropout for regularization

Fully connected layers for classification

Results
The model achieves:

Training accuracy: ~95%

Validation accuracy: ~90%

Test accuracy: ~85%

License
This project uses the Chest X-Ray Images dataset which is available under the CC BY 4.0 license.

Acknowledgments
Data provided by Paul Mooney on Kaggle

Inspired by research in medical image analysis and deep learning
