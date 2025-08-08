
# All imports

# Image capture & processing
import cv2
from PIL import Image
import numpy as np
import os

# ML model & preprocessing
import torch
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity

"""
def capture_image(filename="selfie.jpg"):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    print("Press SPACE to capture...")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            break
    cap.release()
    cv2.destroyAllWindows()
    return filename
"""

def extract_features(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0)

def load_celeb_features(base_dir, gender, model, preprocess):
    celeb_dir = os.path.join(base_dir, gender)
    celeb_features = {}
    for celeb_name in os.listdir(celeb_dir):
        celeb_path = os.path.join(celeb_dir, celeb_name)
        if os.path.isdir(celeb_path):
            features_list = []
            for img_file in os.listdir(celeb_path):
                img_path = os.path.join(celeb_path, img_file)
                features = extract_features(img_path, model, preprocess)
                features_list.append(features)
            celeb_features[celeb_name] = features_list
    return celeb_features

# Python
def find_lookalike(selfie_features, celeb_features_dict):
    best_match = None
    best_score = float('inf')

    for name, feat in celeb_features_dict.items():
        # Ensure feat is a tensor
        if isinstance(feat, list):
            feat = torch.tensor(feat)

        # Ensure feat is 1D
        if feat.dim() > 1:
            feat = feat.squeeze()

        score = torch.norm(selfie_features - feat)
        if score < best_score:
            best_score = score
            best_match = name

    return best_match, best_score.item()

# Python
if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()

    filename = input("Enter the path to your selfie image: ").strip()
    gender = input("Enter your gender (males/females): ").strip().lower()

    celeb_features = load_celeb_features('celeb_images', gender, feature_extractor, preprocess)
    selfie_features = extract_features(filename, feature_extractor, preprocess).squeeze()

    lookalike = find_lookalike(selfie_features, celeb_features)
    print(f"You look like: {lookalike}")