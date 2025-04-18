## 📄 Project Description (for CV)

This project presents a comprehensive deep learning pipeline for **Facial Expression Recognition**, **Facial Landmark Detection**, and **Bounding Box Regression** using a custom multi-task learning architecture built with PyTorch. It reflects expertise in computer vision, deep learning, and practical deployment of multi-output neural networks.

### 🧠 Objective

To design and train a single, efficient deep neural network capable of performing three facial analysis tasks simultaneously:
1. **Facial Expression Classification** – Predict one of seven emotion categories (e.g., Happy, Sad, Angry, etc.).
2. **Landmark Detection** – Predict 478 3D facial landmark coordinates (x, y, z) per image.
3. **Face Localization** – Predict bounding box coordinates for face detection in the form of (x_min, y_min, x_max, y_max).

### 🛠️ Key Features & Technical Contributions

- **Custom Dataset Preparation**: Preprocessed a dataset of 24,000 annotated facial images containing expression labels, 3D landmarks, and bounding box annotations. Applied face cropping based on bounding boxes before model input.
  
- **Model Architecture**:
  - Designed a custom convolutional neural network (CNN) from scratch without pretrained weights.
  - Implemented three output heads:
    - Classification head using softmax for emotion prediction.
    - Regression head for facial landmark coordinates.
    - Regression head for bounding box localization.
  
- **Loss Function Design**:
  - Constructed a multi-task loss:  
    `Total Loss = CrossEntropyLoss + γ * MSELoss (landmarks) + β * SmoothL1Loss (bbox)`
  - Hyperparameters `gamma` and `beta` are tunable for balancing task importance.

- **Training Framework**:
  - Implemented a custom PyTorch training loop with:
    - Early stopping based on validation loss
    - Learning rate scheduling with `ReduceLROnPlateau`
    - Model checkpointing and recovery
    - Real-time loss tracking and logging

- **Data Augmentation**:
  - Used OpenCV and torchvision transforms to apply:
    - Random horizontal flips, rotations, scaling
    - Color jittering for better generalization

- **Evaluation & Visualization**:
  - Evaluated expression accuracy, landmark mean absolute error (MAE), and IoU for bounding boxes.
  - Created visualization tools to overlay landmarks and bounding boxes on input images with predicted emotion labels.

### 📈 Results

- Achieved high performance on validation set with balanced multi-task learning.
- Robust to varying facial poses and lighting conditions due to diverse augmentation.
- Real-time inference capable of displaying bounding boxes, facial landmarks, and expression predictions.

### 🧪 Skills Demonstrated

- PyTorch (custom model design and training)
- Computer Vision (OpenCV, facial analysis, image augmentation)
- Multi-task Learning
- Model Evaluation and Visualization
- Clean code structure and modular project design

### Example
![Figure_1](https://github.com/user-attachments/assets/5c8b0f98-102c-480e-957d-17771ac580d2)

This project showcases the integration of multiple deep learning techniques into a single, unified system—highlighting both theoretical understanding and practical implementation skills in facial recognition systems.
