import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_attention_map(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.vit(image_tensor, output_attentions=True)
        attentions = outputs.attentions[-1]  # Last layer
    attn = attentions[0].mean(0)  # [197, 197]
    cls_attn = attn[0, 1:]  # Exclude CLS
    map_size = int(cls_attn.size(0) ** 0.5)
    cls_attn = cls_attn.reshape(map_size, map_size).cpu().numpy()
    cls_attn = cv2.resize(cls_attn, (224, 224))
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    return cls_attn

def evaluate(model, dataloader, device, class_names, feature_extractor, save_dir=None):
    model.eval()
    all_preds, all_labels = [], []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1).cpu()
            labels = labels.cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

            for i in range(len(preds)):
                if preds[i] != labels[i] and save_dir:
                    img_tensor = images[i].cpu()
                    img_pil = to_pil_image(img_tensor)
                    pred_label = class_names[preds[i]]
                    true_label = class_names[labels[i]]
                    base_name = f"{batch_idx}_{i}_pred-{pred_label}_true-{true_label}"
                    img_pil.save(os.path.join(save_dir, f"{base_name}.jpg"))

                    # Save attention overlay
                    attn_map = get_attention_map(model, img_tensor, device)
                    raw_np = np.array(img_pil.resize((224, 224)))
                    heatmap = cv2.applyColorMap(np.uint8(attn_map * 255), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(raw_np, 0.6, heatmap, 0.4, 0)
                    cv2.imwrite(os.path.join(save_dir, f"{base_name}_attn.jpg"), overlay)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"📊 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1

def main():
    # === Paths ===
    train_path = "content/Task_A/train"
    val_path = "/content/Task_A/val"
    model_save_path = "/content/best_finetuned"
    mislead_save_path = "results/misclassified_train_val"

    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=5  # <-- Change to your number of classes
    ).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = train_dataset.classes

    # === Optimizer ===
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    best_f1 = 0.0

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f"\n📦 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"📉 Train Loss: {running_loss / len(train_loader):.4f}")

        # === Validation & Save Misclassified Attention Maps
        acc, prec, rec, f1 = evaluate(model, val_loader, device, class_names, feature_extractor, save_dir=mislead_save_path)

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(model_save_path)
            feature_extractor.save_pretrained(model_save_path)
            print("✅ Best model saved!")

if __name__ == "__main__":
    main()
