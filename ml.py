# import streamlit as st
# import pandas as pd
# import pickle

# # Load trained model
# with open("disaster_relief_model.pkl", "rb") as f:
#     model = pickle.load(f)
    
# st.title("🌍 Disaster Relief Prediction System")
# st.subheader("Real-Time Resource Allocation")

# # UI - Categorical Inputs
# disaster_type = st.selectbox("Disaster Type", ['earthquake', 'flood', 'hurricane'])  # match your data
# location = st.selectbox("Location", ['urban', 'rural'])  # match your data
# severity_level = st.selectbox("Severity Level", ['low', 'medium', 'high'])
# weather_condition = st.selectbox("Weather Condition", ['sunny', 'rainy', 'stormy'])

# # UI - Numerical Inputs
# road_blocked_input = st.selectbox("Road Blocked", ["Yes", "No"])
# road_blocked = 1 if road_blocked_input == "Yes" else 0

# population_affected = st.number_input("Population Affected", min_value=0)
# medical_cases = st.number_input("Medical Cases", min_value=0)
# infrastructure_damage_percent = st.slider("Infrastructure Damage (%)", 0, 100)

# # Prediction button
# if st.button("Predict Relief Requirements"):
#     input_data = pd.DataFrame([{
#         'disaster_type': disaster_type,
#         'location': location,
#         'severity_level': severity_level,
#         'weather_condition': weather_condition,
#         'road_blocked': road_blocked,
#         'population_affected': population_affected,
#         'medical_cases': medical_cases,
#         'infrastructure_damage_percent': infrastructure_damage_percent
#     }])

#     prediction = model.predict(input_data)

#     food, rescue, medical = prediction[0]

#     st.success("📦 Relief Prediction Results:")
#     st.write(f"🍚 Food Required (tons): **{food:.2f}**")
#     st.write(f"🚑 Medical Aid Required: **{medical:.2f}**")
#     st.write(f"🧑‍🚒 Rescue Teams Required: **{rescue:.2f}**")



import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# Load trained model
with open("disaster_relief_model.pkl", "rb") as f:
    model = pickle.load(f)

# Configure Gemini API
genai.configure(api_key="AIzaSyDS9uZKp5l6xOdgqLVUxcxjp8QvDwNY69M")  # Replace with your actual API key

st.title("🌍 Disaster Relief Traditional System")
st.subheader("Real-Time Resource Allocation")

# UI - Inputs
disaster_type = st.selectbox("Disaster Type", ['earthquake', 'flood', 'hurricane'])
location = st.selectbox("Location", ['urban', 'rural'])
severity_level = st.selectbox("Severity Level", ['low', 'medium', 'high'])
weather_condition = st.selectbox("Weather Condition", ['sunny', 'rainy', 'stormy'])
road_blocked_input = st.selectbox("Road Blocked", ["Yes", "No"])
road_blocked = 1 if road_blocked_input == "Yes" else 0
population_affected = st.number_input("Population Affected", min_value=0)
medical_cases = st.number_input("Medical Cases", min_value=0)
infrastructure_damage_percent = st.slider("Infrastructure Damage (%)", 0, 100)

# Predict + Generate Report
if st.button("Predict Relief Requirements"):
    input_data = pd.DataFrame([{
        'disaster_type': disaster_type,
        'location': location,
        'severity_level': severity_level,
        'weather_condition': weather_condition,
        'road_blocked': road_blocked,
        'population_affected': population_affected,
        'medical_cases': medical_cases,
        'infrastructure_damage_percent': infrastructure_damage_percent
    }])

    prediction = model.predict(input_data)
    food, rescue, medical = prediction[0]

    st.success("📦 Relief Prediction Results:")
    st.write(f"🍚 Food Required (tons): **{food:.2f}**")
    st.write(f"🚑 Medical Aid Required: **{medical:.2f}**")
    st.write(f"🧑‍🚒 Rescue Teams Required: **{rescue:.2f}**")

    # Gemini prompt
    prompt = f"""
    You are an expert disaster response analyst helping to prepare formal reports for disaster relief planning.

    Generate a professional, detailed report based on the following data:

    Disaster Information:
    - Type: {disaster_type}
    - Location: {location}
    - Severity Level: {severity_level}
    - Weather Condition: {weather_condition}

    Impact Assessment:
    - Road Blocked: {road_blocked_input}
    - Population Affected: {population_affected}
    - Medical Cases: {medical_cases}
    - Infrastructure Damage: {infrastructure_damage_percent}%

    Predicted Relief Requirements (from ML model):
    - Food Required: {food:.2f} tons
    - Medical Aid Required: {medical:.2f} units
    - Rescue Teams Required: {rescue:.2f}

    Instructions:
    - Explain in detail how each input factor contributes to the predicted resource requirements.
    - Justify the predicted numbers using real-world disaster response reasoning.
    - Include a short summary at the end titled "Recommended Action Plan".
    - Make the report easy to present to NGOs, government officials, or the media.
    - Use formal, professional tone with clear headers.
    - Do not include AI-related disclaimers or footnotes.
    """

    with st.spinner("🧠 Generating Report with Gemini..."):
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(prompt)
        report_text = response.text

    st.markdown("### 🧾 Generated Report:")
    st.markdown(report_text)

    # --- PDF generation using reportlab ---
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    # Split and add paragraphs
    story = []
    for line in report_text.split('\n'):
        if line.strip():  # Skip empty lines
            story.append(Paragraph(line.strip(), styles["Normal"]))
    doc.build(story)

    # PDF Download Button
    st.download_button(
        label="📥 Download Report as PDF",
        data=buffer.getvalue(),
        file_name="disaster_relief_report.pdf",
        mime="application/pdf"
    )



