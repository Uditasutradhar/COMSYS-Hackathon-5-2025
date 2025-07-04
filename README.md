# COMSYS-Hackathon-5-2025
<br>
Team Name: Little Bits <br>
AUTHOR Udita SUTRADHAR (Leader) <br>
      Arpita Shaw (Member) <br>
      Dr. Showmik Bhowmik (Supervisor)
<br>

Institute: Ghani Khan Chowdhury Institute of Engineering and Technology, Malda

<br>

#Task A 
##🚀 Gender Classification 
<br>
This project classifies gender (male/female) from facial images using the Vision Transformer (`ViT-base-patch16-224-in21k`) fine-tuned on a custom dataset.

---
- 🔢 Type: Binary Classification
- 🧾 Dataset:
  - train: contains 'male' and 'female' folders
  - val: follow the same format
- 🏁 Objective: Predict gender from unseen face images

---

## 🏗️ Model

- Pretrained `ViT-Base-Patch16-224-IN21K`
- Fine-tuned using AdamW optimizer with:
  - LR: 2e-5
  - Label Smoothing: 0.1
  - Augmentations: RandomResizedCrop, ColorJitter, HorizontalFlip

---

## 🏃 Training Pipeline

- ✅ Checkpointing per epoch
- ✅ Best model saved as 'best_finetuned'
- ✅ Metrics computed: Accuracy, Precision, Recall, F1-score

---

## 📊 Final Evaluation Results

On the Val set:

- ✅ Accuracy : 0.9739
- 🎯 Precision: 0.9722
- 🔁 Recall   : 0.9937
- 📌 F1 Score : 0.9828


---

## 🔬 Explain ability

We visualize the 'attention maps' of misclassified images to interpret the model’s decisions:

![Attention Example](./attention_overlay.jpg)

> Attention maps generated using ViT’s final-layer self-attention and overlayed using OpenCV.

---

## ❗ Error Analysis

We save all misclassified test images in `misclassified_test_images/` with filenames like:3_1_pred-female_true-male.jpg
