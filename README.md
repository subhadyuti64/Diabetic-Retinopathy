# 🩺 Diabetic Retinopathy Diagnostic App

A Streamlit-based AI-powered diagnostic tool that:
- Classifies retinal fundus images using a fine-tuned vision transformer (ViT)
- Uses Google's Gemini (via LangChain) to generate detailed, patient-friendly medical reports
- Exports the final report as a downloadable PDF

---

## 🚀 Features

✅ Upload retinal images (JPG/PNG)  
✅ Automatically classify diabetic retinopathy into 5 stages:  
`No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative DR`  
✅ Input required patient information: Name, Age, Gender, Date  
✅ Generate a detailed diagnostic report using **Gemini 2.0 Flash**  
✅ Download the report as a **PDF**  
✅ Privacy-focused: runs locally without uploading images externally

---

## 🧠 Tech Stack

- **Frontend/UI**: Streamlit
- **Image Classification**: PyTorch + HuggingFace Transformers
- **LLM Reporting**: LangChain + Google Gemini (GenerativeAI)
- **PDF Generation**: FPDF
- **Environment Management**: Python virtual environment (`venv`)
- **Credential Handling**: `python-dotenv`

---

## 📸 Sample Output

> Prediction: **Moderate Diabetic Retinopathy**

> ✅ Automatically generates a detailed 1–2 page PDF explaining:
- The diagnosis
- Causes
- Symptoms
- Urgency & next steps
- Lifestyle/diet suggestions

---

## 🛠️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/subhadyuti64/Diabetic-Retinopathy.git


# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Google API credentials
# Put your service account JSON in this folder and add this to .env:
echo "GOOGLE_API_KEY=your_api_key_here" >> .env
echo "GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your/credentials.json" >> .env

# 5. Run the main.ipynb file to get the model.pth and processor.pth

# 6. Run the app
streamlit run test.py
