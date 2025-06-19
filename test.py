import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
import os
import io
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification,ViTForImageClassification
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
from datetime import date
from fpdf import FPDF
from io import BytesIO

load_dotenv()

# Load model and processor
model = torch.load("model.pth", weights_only=False)
processor = torch.load("processor.pth", weights_only=False)

# Gemini setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt template
report_prompt = PromptTemplate(
    input_variables=["name", "age", "gender", "date", "prediction"],
    template="""
You are a senior ophthalmologist preparing a diagnostic report for a patient.
The Heading of the report should be: DIAGNOSIS REPORT
### Patient Information:
- **Name**: {name}
- **Age**: {age}
- **Gender**: {gender}
- **Date of Report**: {date}

### Diagnosis:
The AI system has classified the patient's condition as **{prediction} Diabetic Retinopathy**.

Please generate a detailed and patient-friendly medical report including the following sections:

1. **Understanding {prediction} Diabetic Retinopathy**
2. **Possible Causes and Progression**
3. **Potential Symptoms**
4. **Urgency and Recommended Next Steps**
5. **Lifestyle and Dietary Suggestions**

Make the explanation medically accurate, compassionate, and easy to understand. Use bullet points and bold section headers where appropriate. Avoid unnecessary medical jargon.
"""
)

# Chain
report_chain: Runnable = report_prompt | llm

def generate_pdf_report(patient_name, report_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.multi_cell(0, 10, txt=report_data)

    # Get PDF content as a string
    pdf_string = pdf.output(dest='S').encode('latin1')

    pdf_buffer = BytesIO()
    pdf_buffer.write(pdf_string)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# --- Streamlit App UI ---
st.title("🩺 Diabetic Retinopathy Diagnostic App")

#  Patient Info
st.subheader("👤 Patient Information")
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=1, max_value=120)
patient_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
report_date = st.date_input("Date of Report", value=date.today())

#  Validation
patient_info_valid = (
    patient_name.strip() != "" and
    patient_age > 0 and
    patient_gender != "Select"
)

if not patient_info_valid:
    st.warning("Please fill in all patient details to proceed.")

#  Image Upload
st.subheader("📤 Upload Retinal Image")
image = st.file_uploader("Upload a retinal scan image", type=["jpg", "jpeg", "png"])

#  Classification
if image is not None:
    image_display = Image.open(image)
    st.image(image_display, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify Retinal Scan", disabled=not patient_info_valid):
        with st.spinner("Classifying..."):
            inputs = processor(image_display, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
            result = labels[predicted_label]
            st.session_state["dr_prediction"] = result
            st.success(f"Prediction: {result} Diabetic Retinopathy")

#  Generate Medical Report
if "dr_prediction" in st.session_state:
    if st.button("📄 Generate Medical Report", disabled=not patient_info_valid):
        with st.spinner("Generating report with Gemini..."):
            report = report_chain.invoke({
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "date": report_date.strftime("%Y-%m-%d"),
                "prediction": st.session_state["dr_prediction"]
            })
            st.markdown("### 📝 Diagnostic Report")
            st.markdown(report.content, unsafe_allow_html=True)

            # Save for download
            st.session_state["final_report"] = report.content

#  Download Button
if "final_report" in st.session_state:
    pdf_buffer = generate_pdf_report(patient_name, st.session_state["final_report"])
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name=f"{patient_name}_DR_Report.pdf",
        mime="application/pdf"
    )
