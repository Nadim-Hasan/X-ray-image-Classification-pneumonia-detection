Chest X-Ray Pneumonia Detection using Deep Learning
A comprehensive deep learning project for classifying chest X-ray images into Normal and Pneumonia categories using convolutional neural networks.

📋 Project Overview
This project implements a computer vision solution to detect pneumonia from chest X-ray images. The model uses transfer learning with a CNN architecture to classify medical images with high accuracy, potentially assisting healthcare professionals in diagnosis.

🗃️ Dataset Information
Source: Kaggle Chest X-Ray Images (Pneumonia) Dataset

Dataset Statistics
Total Images: 5,856 validated X-ray images

Classes:

Normal: 1,583 images

Pneumonia: 4,273 images

Image Format: JPEG

Image Dimensions: Various sizes (typically 1024×1024 or 1024×1280)

Dataset Structure
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/

⚙️ Installation & Setup
Prerequisites
Python 3.7+

GPU support recommended (CUDA compatible)

Step-by-Step Installation
Clone the repository
git clone https://github.com/Nadim-Hasan/X-ray-image-Classification-pneumonia-detection.git
cd X-ray-image-Classification-pneumonia-detection

Install required dependencies

bash - pip install -r requirements.txt

Set up Kaggle API (for dataset download)

Create a Kaggle account at https://www.kaggle.com

Go to Account → Create API Token to download kaggle.json

Place kaggle.json in the project directory

🚀 Usage
Data Download
# Download dataset from Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract the dataset
unzip chest-xray-pneumonia.zip

Model Training
python
# Run the training notebook
jupyter notebook Xray_image_classification.ipynb

Making Predictions
python
from prediction import predict_image

# Load and predict on a new X-ray image
result = predict_image('path/to/your/xray_image.jpg')
print(f"Prediction: {result['class']} with {result['confidence']:.2f}% confidence")
🧠 Model Architecture
The project utilizes a convolutional neural network with the following structure:

Base Model: Pre-trained CNN (VGG16/ResNet50) for feature extraction

Custom Layers:

Global Average Pooling

Dense layers with Dropout regularization

Final sigmoid activation for binary classification

Image Preprocessing: Resizing, normalization, and data augmentation

📊 Results & Performance
Training Metrics
Training Accuracy: ~95%

Validation Accuracy: ~90%

Test Accuracy: ~88%

Evaluation Metrics
Precision: 92%

Recall: 89%

F1-Score: 90%

AUC: 0.96

📁 Project Structure
text
Xray_image_classification/
├── data/
│   └── chest_xray/          # Dataset directory
├── models/                  # Saved model weights
├── utils/                   # Utility functions
├── Xray_image_classification.ipynb  # Main notebook
├── requirements.txt         # Python dependencies
├── kaggle.json             # Kaggle API credentials (gitignored)
└── README.md               # Project documentation
🏥 Medical Disclaimer
This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.

📜 License
This project uses the Chest X-Ray Images dataset which is available under the Creative Commons Attribution 4.0 International License.

The code in this repository is released under the MIT License.

🙏 Acknowledgments
Data provided by Paul Mooney on Kaggle

Inspiration from various research papers on medical image analysis

Computing resources provided by Google Colab

📫 Contact
For questions or suggestions about this project, please contact:

Name: Nadim Hasan

Email: [Your Email]

GitHub: Nadim-Hasan
