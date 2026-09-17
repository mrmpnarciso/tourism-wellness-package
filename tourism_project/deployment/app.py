
import os
import joblib
import pandas as pd
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

HF_USERNAME = "mrmpnarciso"
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-package-model"
HF_TOKEN = os.getenv("HF_TOKEN")

model_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    filename="best_model.joblib",
    token=HF_TOKEN,
)
model = joblib.load(model_path)


@spaces.GPU
def predict(age, type_of_contact, city_tier, occupation, gender, num_persons_visiting,
            preferred_property_star, marital_status, num_trips, passport, own_car,
            num_children_visiting, designation, monthly_income, pitch_satisfaction_score,
            product_pitched, num_followups, duration_of_pitch):

    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        return (
            f'<div style="background:#EEAF9D; color:#182335; padding:24px; '
            f'border-radius:10px; text-align:center; font-size:22px; font-weight:700;">'
            f'Likely to purchase<br><span style="font-size:16px; font-weight:400;">'
            f'Confidence: {probability:.1%}</span></div>'
        )
    else:
        return (
            f'<div style="background:#E15546; color:#EAE4CC; padding:24px; '
            f'border-radius:10px; text-align:center; font-size:22px; font-weight:700;">'
            f'Unlikely to purchase<br><span style="font-size:16px; font-weight:400;">'
            f'Confidence: {1 - probability:.1%}</span></div>'
        )


theme = gr.themes.Base().set(
    body_background_fill="#182335",
    body_background_fill_dark="#182335",
    background_fill_primary="#182335",
    background_fill_primary_dark="#182335",
    background_fill_secondary="#1f2d45",
    background_fill_secondary_dark="#1f2d45",
    block_background_fill="#1f2d45",
    block_background_fill_dark="#1f2d45",
    block_border_width="0px",
    block_border_width_dark="0px",
    body_text_color="#EAE4CC",
    body_text_color_dark="#EAE4CC",
    body_text_color_subdued="#EAE4CC",
    block_label_text_color="#EEAF9D",
    block_label_text_color_dark="#EEAF9D",
    input_background_fill="#0f1826",
    input_background_fill_dark="#0f1826",
    input_border_width="0px",
    input_border_width_dark="0px",
    button_primary_background_fill="#E15546",
    button_primary_background_fill_hover="#c2402f",
    button_primary_text_color="#EAE4CC",
    slider_color="#E15546",
    slider_color_dark="#E15546",
)

custom_css = """
.gradio-container { color-scheme: dark !important; background: #182335 !important; }
#title { text-align: center; color: #EAE4CC !important; font-weight: 700 !important; }
#subtitle { text-align: center; color: #EEAF9D !important; font-size: 16px; margin-bottom: 20px; }
label span, .block label, .block label span { color: #EEAF9D !important; font-weight: 600 !important; }
input[type="text"], input[type="number"], textarea, select {
  background: #0f1826 !important;
  color: #EAE4CC !important;
  border: none !important;
}
.block { background: #1f2d45 !important; border: none !important; box-shadow: none !important; }
button[role="radio"] { color: #EAE4CC !important; background: #0f1826 !important; border: none !important; }
button[role="radio"][aria-checked="true"] { background: #E15546 !important; color: #EAE4CC !important; }
button.primary, .primary { background: #E15546 !important; color: #EAE4CC !important; border: none !important; }
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("# Wellness Tourism Package: Purchase Prediction", elem_id="title")
    gr.Markdown(
        "Fill in the customer's details to predict whether they're likely to buy the Wellness Tourism Package.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=35)
            type_of_contact = gr.Dropdown(["Self Enquiry", "Company Invited"], label="Type of Contact")
            city_tier = gr.Dropdown([1, 2, 3], label="City Tier")
            occupation = gr.Dropdown(["Salaried", "Free Lancer", "Small Business", "Large Business"], label="Occupation")
            gender = gr.Dropdown(["Male", "Female"], label="Gender")
            num_persons_visiting = gr.Number(label="Number of Persons Visiting", value=2)

        with gr.Column():
            preferred_property_star = gr.Dropdown([3.0, 4.0, 5.0], label="Preferred Property Star")
            marital_status = gr.Dropdown(["Single", "Married", "Divorced"], label="Marital Status")
            num_trips = gr.Number(label="Number of Trips (per year)", value=2)
            passport = gr.Radio(["Yes", "No"], label="Holds Passport")
            own_car = gr.Radio(["Yes", "No"], label="Owns Car")
            num_children_visiting = gr.Number(label="Number of Children Visiting", value=0)

        with gr.Column():
            designation = gr.Dropdown(["Executive", "Manager", "Senior Manager", "AVP", "VP"], label="Designation")
            monthly_income = gr.Number(label="Monthly Income", value=20000)
            pitch_satisfaction_score = gr.Slider(1, 5, value=3, step=1, label="Pitch Satisfaction Score")
            product_pitched = gr.Dropdown(["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], label="Product Pitched")
            num_followups = gr.Number(label="Number of Followups", value=3)
            duration_of_pitch = gr.Number(label="Duration of Pitch (minutes)", value=10)

    submit_btn = gr.Button("Predict", variant="primary", size="lg")
    result = gr.HTML()

    submit_btn.click(
        fn=predict,
        inputs=[age, type_of_contact, city_tier, occupation, gender, num_persons_visiting,
                preferred_property_star, marital_status, num_trips, passport, own_car,
                num_children_visiting, designation, monthly_income, pitch_satisfaction_score,
                product_pitched, num_followups, duration_of_pitch],
        outputs=result,
    )

if __name__ == "__main__":
    demo.launch()
