import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# --- CONFIG ---
CELEB_BASE_DIR = "celeb_images"  # Structure: celeb_images/gender/celeb_name/*.jpg
# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY environment variable not set!")

# Set up Groq-compatible OpenAI API client
client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
)

# Generate an explanation using a supported model
def generate_explanation(celeb_name, score):
    prompt = (
        f"Write a mid length, fun, and friendly but not so friendly kiss ass explanation why someone matches {celeb_name} "
        f"with a similarity score of {score:.2f}. Mention all aspects that caused this similarity to occur like face structure,hair or vibe "
        f"(include other things on your own) and give them tips on what to improve nicely, the limit is 5000 tokens."
    )
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # ✅ updated to a supported Groq model
            messages=[
                {"role": "system", "content": "You are a witty celebrity lookalike commentator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5000,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"🧠 Groq failed: {e}"


def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# --- Load feature extractor model ---
@st.cache_resource
def load_model():
    resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
    feature_extractor.eval()
    return feature_extractor

# --- Get celeb list ---
def get_celebrity_list(gender):
    celeb_dir = os.path.join(CELEB_BASE_DIR, gender)
    return sorted([d for d in os.listdir(celeb_dir) if os.path.isdir(os.path.join(celeb_dir, d))])

# --- Prediction ---
def predict_lookalike(model, user_img_tensor, celeb_list, gender):
    max_score = -1
    best_match = None

    with torch.no_grad():
        user_vec = model(user_img_tensor).squeeze().view(-1)

        for celeb_name in celeb_list:
            celeb_img_dir = os.path.join(CELEB_BASE_DIR, gender, celeb_name)
            celeb_imgs = [f for f in os.listdir(celeb_img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

            for img_file in celeb_imgs:
                celeb_img_path = os.path.join(celeb_img_dir, img_file)
                celeb_img = Image.open(celeb_img_path).convert("RGB")
                celeb_tensor = preprocess_image(celeb_img)
                celeb_vec = model(celeb_tensor).squeeze().view(-1)

                score = torch.nn.functional.cosine_similarity(user_vec, celeb_vec, dim=0).item()

                if score > max_score:
                    max_score = score
                    best_match = celeb_name

    return best_match, max_score

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Celebrity Lookalike", page_icon="🌟", layout="centered")
    st.title("🌟 Celebrity Lookalike Matcher")
    st.markdown("Upload your selfie and discover which celebrity you resemble!")

    gender = st.selectbox("Who do you want to be matched with?", ["males", "females"])

    uploaded_file = st.file_uploader("Upload your selfie (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        user_img = Image.open(uploaded_file).convert("RGB")
        st.image(user_img, caption="👤 Your Uploaded Image", width=300)
        user_tensor = preprocess_image(user_img)

        if st.button("✨ Start Matching"):
            with st.spinner("Analyzing your features and searching for your celebrity twin..."):
                model = load_model()
                celeb_list = get_celebrity_list(gender)
                lookalike, score = predict_lookalike(model, user_tensor, celeb_list, gender)

            if lookalike:
                st.success(f"🎉 You look like: **{lookalike}** (Similarity Score: {score:.2f})")

                # 🖼️ Show celeb images
                celeb_img_dir = os.path.join(CELEB_BASE_DIR, gender, lookalike)
                celeb_imgs = [os.path.join(celeb_img_dir, img) for img in os.listdir(celeb_img_dir)]
                st.image([Image.open(img) for img in celeb_imgs], caption=[lookalike] * len(celeb_imgs), width=200)

                st.markdown("---")  # Horizontal rule for separation

                # 🎨 Explanation from OpenAI
                with st.spinner("Generating explanation ..."):
                    explanation = generate_explanation(lookalike, score)
                st.markdown("### 🤖 Why this match?")
                st.write(explanation)

if __name__ == "__main__":
    main()

