 🎭 Celebrity Lookalike Finder


 🚀 Features
- 📸 **Face Matching** — Compares your photo against a curated set of celebrity images.
- 🧠 ** AI Commentary** — Powered by Groq’s `llama3-70b-8192` for natural explanations.
- 📂 **Organized Celebrity Database** — Sorted by gender for easy scaling and maintenance.
- 🌐 **Simple Interface** — Built with Streamlit for a clean and interactive experience.
- ⚡ **Ultra-Fast API Calls** — Groq delivers low latency and high token capacity.



## ⚙️ Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/celebrity-lookalike.git
cd celebrity-lookalike
```
2. **Create and activate a virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```
3. **Install dependencies**
```bash
pip install -r reqs.txt
```
4. **Set up your `.env` file**
```env
GROQ_API_KEY=your_groq_api_key_here
```

## ▶️ Run the App
```bash
streamlit run interface.py
```
Then open the provided URL in your browser.

## 🔑 API Information
This app uses **Groq’s API** in OpenAI-compatible mode:
- **Base URL**: `https://api.groq.com/openai/v1`
- **Model**: `llama3-70b-8192`
- **Max Tokens**: Up to 8192 per request

## 📜 License
MIT License – free for personal and commercial use.

---
**Author:** Ali Khatib
