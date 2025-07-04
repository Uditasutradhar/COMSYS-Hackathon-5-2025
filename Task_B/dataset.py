import random # Import the random module
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


class FaceVerificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_negative_pairs=1):
        self.transform = transform
        self.pairs = []
        split_path = os.path.join(root_dir, split)
        person_folders = [p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))]

        # Positive pairs
        for person in person_folders:
            person_path = os.path.join(split_path, person)
            clear_image = next((f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)
            distortion_path = os.path.join(person_path, 'distortion')
            if clear_image and os.path.exists(distortion_path):
                clear_img_path = os.path.join(person_path, clear_image)
                for dist_file in os.listdir(distortion_path):
                    if dist_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        dist_img_path = os.path.join(distortion_path, dist_file)
                        self.pairs.append((clear_img_path, dist_img_path, 1))

        # Negative pairs
        for _ in range(len(self.pairs) * num_negative_pairs):
            person_a, person_b = random.sample(person_folders, 2)
            path_a = os.path.join(split_path, person_a)
            path_b = os.path.join(split_path, person_b)
            img_a = next((f for f in os.listdir(path_a) if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)
            img_b = next((f for f in os.listdir(path_b) if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)
            if img_a and img_b:
                self.pairs.append((os.path.join(path_a, img_a), os.path.join(path_b, img_b), 0))
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label
