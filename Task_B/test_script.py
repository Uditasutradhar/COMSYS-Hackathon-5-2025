import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# ✅ Define the model
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super(FaceEmbeddingModel, self).__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(base.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

# ✅ Evaluation function
def evaluate_face_verification(test_path, model_path, threshold=0.7):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    save_results_to = os.path.join(test_path, "face_verification_results.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FaceEmbeddingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Build reference embeddings (actual images)
    person_embeddings = {}
    for person in os.listdir(test_path):
        person_path = os.path.join(test_path, person)
        if not os.path.isdir(person_path):
            continue
        reference_images = []
        for f in os.listdir(person_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'distort' not in f.lower():
                img_path = os.path.join(person_path, f)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img_tensor).cpu()
                reference_images.append(emb)
        if reference_images:
            person_embeddings[person] = torch.stack(reference_images).mean(dim=0)

    # Evaluate distorted images
    all_preds, all_labels = [], []
    results = []
    for person in os.listdir(test_path):
        distorted_path = os.path.join(test_path, person, 'distortion')
        if not os.path.exists(distorted_path):
            continue
        for f in os.listdir(distorted_path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(distorted_path, f)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                test_emb = model(img_tensor).cpu()

            best_sim = -1
            best_match = None
            for ref_person, ref_emb in person_embeddings.items():
                sim = F.cosine_similarity(test_emb, ref_emb).item()
                if sim > best_sim:
                    best_sim = sim
                    best_match = ref_person

            is_match = 1 if best_match == person and best_sim > threshold else 0
            true_label = 1 if best_match == person else 0
            all_preds.append(is_match)
            all_labels.append(true_label)
            results.append({
                "test_image": img_path,
                "true_person": person,
                "predicted_person": best_match,
                "similarity": round(best_sim, 4),
                "predicted_label": is_match,
                "true_label": true_label
            })

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n📊 Face Verification Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    os.makedirs(os.path.dirname(save_results_to), exist_ok=True)
    with open(save_results_to, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "details": results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {save_results_to}")

# ✅ MODIFY THIS ONLY
test_path = "/content/drive/MyDrive/AI_demos/Comys_Hackathon5/Task_B/test"
model_path = "/content/drive/MyDrive/AI_demos/Comys_Hackathon5/Task_B/face_verification.pth"

# ✅ Run
evaluate_face_verification(test_path, model_path)

