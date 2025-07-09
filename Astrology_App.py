import os
import gradio as gr
import google.generativeai as genai
from datetime import datetime

# Configure Gemini API
genai.configure(api_key="AIzaSyDrx7j5XPIztMz274t-ItjxSE55sr6Q39g")
model = genai.GenerativeModel('gemini-2.5-flash')

# Prediction logic
def get_astrology_prediction(name, dob, zodiac, question):
    try:
        dob_obj = datetime.strptime(dob, "%Y-%m-%d")
        prompt = f"""
        You are a wise astrologer.
        Name: {name}
        DOB: {dob_obj.strftime('%B %d, %Y')}
        Zodiac: {zodiac}
        Question: {question}

        Provide a personalized astrological prediction.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI with full styling
with gr.Blocks(css="""
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body {
    background: #1e1e2f;
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
}

.gradio-container {
    max-width: 750px;
    margin: auto;
    padding: 20px;
    background: #2e2e3e;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
}

h1, label {
    color: #ffcc70 !important;
    font-weight: 600;
}

textarea, input, select {
    background-color: #1e1e2f !important;
    color: #ffffff !important;
    border: 1px solid #555 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    font-size: 15px !important;
}

textarea::placeholder, input::placeholder {
    color: #999 !important;
}

button {
    background-color: #6e40c9;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 600;
    padding: 12px 20px;
    border-radius: 10px;
    cursor: pointer;
    transition: 0.3s ease;
}

button:hover {
    background-color: #8e5ee0;
    transform: scale(1.03);
}
""") as demo:
    gr.Markdown("<h1>🔮 Astrology Predictor</h1>")
    gr.Markdown("Enter your details to get a Gemini-powered astrological prediction.")

    name = gr.Textbox(label="👤 Name", placeholder="e.g., Alice")
    dob = gr.Textbox(label="📅 Date of Birth (YYYY-MM-DD)", placeholder="e.g., 1990-05-12")
    zodiac = gr.Dropdown(
        ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"],
        label="🌌 Zodiac Sign"
    )
    question = gr.Textbox(label="❓ Your Question", lines=3, placeholder="e.g., Will I travel abroad this year?")
    output = gr.Textbox(label="🧙‍♂️ Gemini's Prediction", lines=8)

    predict_btn = gr.Button("✨ Get Prediction")
    predict_btn.click(fn=get_astrology_prediction,
                      inputs=[name, dob, zodiac, question],
                      outputs=output)

demo.launch()
