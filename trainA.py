import os
import torch
import numpy as np
import cv2
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_attention_map(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.vit(image_tensor, output_attentions=True)
        attentions = outputs.attentions[-1]  # Last layer

    attn = attentions[0].mean(0)  # [197, 197]
    cls_attn = attn[0, 1:]  # Drop CLS token

    map_size = int(cls_attn.size(0) ** 0.5)
    cls_attn = cls_attn.reshape(map_size, map_size).cpu().numpy()
    cls_attn = cv2.resize(cls_attn, (224, 224))
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

    return cls_attn

def evaluate(model, dataloader, device, class_names, feature_extractor, save_misleads_dir=None):
    model.eval()
    all_preds, all_labels = [], []

    if save_misleads_dir:
        os.makedirs(save_misleads_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1).cpu()
            labels = labels.cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

            for i in range(len(preds)):
                if preds[i] != labels[i] and save_misleads_dir:
                    img_tensor = images[i].cpu()
                    img_pil = to_pil_image(img_tensor)
                    pred_label = class_names[preds[i]]
                    true_label = class_names[labels[i]]

                    # Save original misclassified image
                    base_name = f"{batch_idx}_{i}_pred-{pred_label}_true-{true_label}"
                    img_pil.save(os.path.join(save_misleads_dir, f"{base_name}.jpg"))

                    # Generate and save attention map
                    attn_map = get_attention_map(model, img_tensor, device)
                    raw_np = np.array(img_pil.resize((224, 224)))
                    heatmap = cv2.applyColorMap(np.uint8(attn_map * 255), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(raw_np, 0.6, heatmap, 0.4, 0)
                    cv2.imwrite(os.path.join(save_misleads_dir, f"{base_name}_attn.jpg"), overlay)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("📊 Evaluation Results:")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🔁 Recall   : {rec:.4f}")
    print(f"📌 F1 Score : {f1:.4f}")

    return acc, prec, rec, f1

# 🔁 Run this directly in Colab
# Please update this path to your actual test data location
test_path = "/content/drive/MyDrive/AI_demos/Comys_Hackathon5/Task_A/val" # <-- Update this path
model_path = "/content/drive/MyDrive/vit-taskA-checkpoints/best_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(model_path).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# Run and save misclassified + attention images
evaluate(model, test_loader, device, class_names, feature_extractor, save_misleads_dir="misclassified_test_images")
