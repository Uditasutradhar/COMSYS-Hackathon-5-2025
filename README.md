# COMSYS-Hackathon-5-2025
<br>
Team Name: Little Bits <br>
AUTHOR- Udita SUTRADHAR (Leader) <br>
        Arpita Shaw (Member) <br>
        Dr. Showmik Bhowmik (Supervisor)
<br>
<br>
Institute: Ghani Khan Chowdhury Institute of Engineering and Technology, Malda

<br>

# Project Overview
### Task A: Gender Classification (Binary Classification) <br>
-Objective: Train a model to accurately classify the gender of a face image.<br>
-Evaluation Metrics: Accuracy | Precision | Recall | F1-Score

### Task B: Face Recognition (Multi-class Classification) <br>
-Objective: Assign each face image to a correct person identity from a known set of individuals.<br>
-Evaluation Metrics: Accuracy | Precision | Recall | F1-Score

#### UPLOAD THE MODEL AND DATASET IN YOUR GOOGLE DRIVE, MOUNT DRIVE TO GOOGLE COLAB
#### RUN THE TEST CODE IN GOOGLE COLAB, JUST MODIFY MODEL AND DATASET PATH

## 🚀 Gender Classification 
<br>
This project classifies gender (male/female) from facial images using the Vision Transformer (`ViT-base-patch16-224-in21k`) fine-tuned on a train dataset.

---
- 🔢 Type: Binary Classification
- 🧾 Dataset:
  - train: contains 'male' and 'female' folders
  - val: follow the same format
- 🏁 Objective: Predict gender from unseen face images

---

#### 🏗️ Model <br>
- Pretrained `ViT-Base-Patch16-224-IN21K`
- Fine-tuned using AdamW optimizer with:
  - LR: 2e-5
  - Label Smoothing: 0.1
  - Augmentations: RandomResizedCrop, ColorJitter, HorizontalFlip

---

#### 🏃 Training Pipeline

- ✅ Checkpointing per epoch
- ✅ Best model saved as 'best_finetuned'
- ✅ Metrics computed: Accuracy, Precision, Recall, F1-score

---

#### 📊 Final Evaluation Results

On the Val set:

- ✅ Accuracy : 0.9739
- 🎯 Precision: 0.9722
- 🔁 Recall   : 0.9937
- 📌 F1 Score : 0.9828


---

🔬 Explain ability

We visualize the 'attention maps' of misclassified images to interpret the model’s decisions:
![Attention Example](./attention_overlay.jpg)
> Attention maps generated using ViT’s final-layer self-attention and overlayed using OpenCV.
#### ❗ Error Analysis
We save all misclassified test images in `misclassified_test_images/` with filenames like:3_1_pred-female_true-male.jpg

---
# 🎭 Face Recognition <br>

---
- 🔢 Type: Multi-class Classification
- 🧾 Dataset:
  dataset/
├── person_1/ <br>
│   ├── image1.jpg  (actual clear image) <br>
│   └── distortion/ <br>
│       ├── distort1.jpg <br>
│       └── distort2.jpg <br>
├── person_2/ <br>
│   ├── image1.jpg <br>
│   └── distortion/ <br>
│       └── distort1.jpg <br>
└── ... <br>

- 🏁Objective :-
- Match an input face image to its corresponding person folder.
-If a test image (or its distorted variant) matches any image in the same folder, it is considered a match (label = 1). 
-If it matches an image from a different folder, it's a non-match (label = 0).

---
📊 Face Verification Results: (on Val Set)
- ✅ Accuracy : 1.0000
- 🎯 Precision: 1.0000
- 🔁 Recall   : 1.0000
- 📌 F1 Score : 1.0000
---

#### 🏗️ Model <br>
-Backbone: ResNet-50 (pretrained on ImageNet) <br>
-Head: Fully connected layer → L2 normalized embedding (512-d) <br>
-Loss: Contrastive Loss for similarity learning <br>
-Evaluation: Cosine similarity with threshold (default: 0.7) <br>

---
#### 📦 Pretrained Model
Path: models/face_verification.pth <br>
Format: PyTorch state_dict <br> 
Size: ~95MB <br>
Backbone: ResNet-50 <br>
Embedding Dim: 512 <br>

---


## 📊 Overall Performance (Val Set) 🔥 <br>

### 📊 Overall Performance (Val Set)

| Metric     | Task A       | Task B       | Overall       |
|------------|--------------|--------------|---------------|
| Accuracy   | 0.9739       | 1.0000       | **0.9869**    |
| Precision  | 0.9722       | 1.0000       | **0.9861**    |
| Recall     | 0.9937       | 1.0000       | **0.9968**    |
| F1-Score   | 0.9828       | 1.0000       | **0.9914**    | 

---
## 🚨 Troubleshooting
<br>
❌ CUDA Out of Memory <br>
-Reduce batch_size in training (default: 32) <br>
-Use a smaller model or lower image resolution <br>
<br>
📁 Dataset Path Issues <br>
-Ensure your dataset is organized as per format above<br>
-Check for missing distortion/ folders <br>
-Use absolute paths for loading
<br>
⚙️ Missing Dependencies <br>
Install all requirements~
<br>

---
### The Pre - requisite <br>
DOWNLOAD AND UPLOAD THE MODEL IN YOUR GOOGLE DRIVE <br>
MAKE SURE TO HAVE THE DATA SET IN YOUR GOOGLE DRIVE <br>

from google.colab import drive <br>
drive.mount('/content/drive')

<br>
mount your drive 
<br>

Run the test <br>

CHANGE THE MODEL AND TEST PATH <br>
<br>
<br>
Thankyou!🙏

---
## 📜 License
This project is developed for COMSYS Hackathon-5,2025 Competition. 


