import os, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import FaceVerificationDataset
from model import FaceEmbeddingModel, ContrastiveLoss
import torch.optim as optim

def train(data_path, save_path="models/face_verification.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = FaceVerificationDataset(root_dir=data_path, split='train', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FaceEmbeddingModel().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(2):
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/2")
        for img1, img2, label in pbar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1), model(img2)
            loss = criterion(out1, out2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1}: Avg Loss = {running_loss / len(loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
