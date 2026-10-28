import os
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Load Dataset & Train Random Forest Model
# ==========================================
heart_data = pd.read_csv("Heart-disease-pridiction.csv")

X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=2)
model.fit(X_train, Y_train)

# ==========================================
# 2. Validation & Prediction Function
# ==========================================
def analyze_heart_risk(age, sex, fbs, trestbps, chol, thalach, exang, cp, restecg, oldpeak, slope, ca, thal):
    # Strict check to ensure no unselected fields
    unselected_inputs = [
        age is None or age == "",
        sex is None or sex == "",
        fbs is None or fbs == "",
        trestbps is None or trestbps == "",
        chol is None or chol == "",
        thalach is None or thalach == "",
        exang is None or exang == "",
        cp is None or cp == "Select Pain Type..." or cp == "",
        restecg is None or restecg == "Select ECG Result..." or restecg == "",
        oldpeak is None or oldpeak == "",
        slope is None or slope == "Select Slope..." or slope == "",
        ca is None or ca == "Select Major Vessels..." or ca == "",
        thal is None or thal == "Select Thalassemia Score..." or thal == ""
    ]

    if any(unselected_inputs):
        return """
        <div style="background: #3a1518; border-radius: 16px; padding: 25px; border: 2px solid #ff7675; text-align: center; color: #ff7675;">
            <h2 style="font-size: 22px; margin: 0 0 10px 0;">⚠️ INCOMPLETE INPUT DETAILS</h2>
            <p style="font-size: 15px; color: #fab1a0; margin: 0;">Please fill in all numerical fields and choose options for all radio buttons and dropdowns before running the assessment.</p>
        </div>
        """

    def parse_dropdown(val):
        try:
            return int(str(val).split(":")[0].split(" ")[0])
        except (ValueError, IndexError):
            return 0

    age_v = float(age)
    trestbps_v = float(trestbps)
    chol_v = float(chol)
    thalach_v = float(thalach)
    oldpeak_v = float(oldpeak)

    sex_v = 1 if "Male" in sex else 0
    fbs_v = 1 if "Yes" in fbs else 0
    exang_v = 1 if "Yes" in exang else 0

    cp_v = parse_dropdown(cp)
    restecg_v = parse_dropdown(restecg)
    slope_v = parse_dropdown(slope)
    ca_v = parse_dropdown(ca)
    thal_v = parse_dropdown(thal)

    input_df = pd.DataFrame([{
        'age': age_v, 'sex': sex_v, 'cp': cp_v, 'trestbps': trestbps_v,
        'chol': chol_v, 'fbs': fbs_v, 'restecg': restecg_v, 'thalach': thalach_v,
        'exang': exang_v, 'oldpeak': oldpeak_v, 'slope': slope_v,
        'ca': ca_v, 'thal': thal_v
    }])

    prob = float(model.predict_proba(input_df)[0][1])
    risk_pct = round(prob * 100, 1)

    if prob < 0.35:
        status_color = "#00b894"
        status_title = "LOW RISK - OPTIMAL CARDIAC HEALTH"
        badge_bg = "#003b2b"
        recommendation = "Patient displays normal cardiovascular indicators. Recommend routine annual checkups and maintaining active lifestyle."
    elif 0.35 <= prob <= 0.65:
        status_color = "#fdcb6e"
        status_title = "MODERATE RISK - CLINICAL MONITORED"
        badge_bg = "#4d3800"
        recommendation = "Elevated risk markers observed. Advise lifestyle modifications, lipid panel monitoring, and secondary evaluation."
    else:
        status_color = "#ff7675"
        status_title = "HIGH RISK - CARDIAC CONSULTATION REQUIRED"
        badge_bg = "#4a0e0e"
        recommendation = "Critical cardiovascular indicators present. Urgent specialist evaluation and cardiac workup recommended."

    risk_factors = []
    if trestbps_v > 140: risk_factors.append("Hypertension (BP > 140)")
    if chol_v > 240: risk_factors.append("Hypercholesterolemia (Chol > 240)")
    if exang_v == 1: risk_factors.append("Exercise Induced Angina")
    if oldpeak_v > 2.0: risk_factors.append("Significant ST Depression")
    if ca_v > 0: risk_factors.append(f"{ca_v} Major Vessel Blockage(s)")

    risk_factors_str = ", ".join(risk_factors) if risk_factors else "No isolated physical risk alerts detected."

    result_html = f"""
    <div style="background: #1e272e; border-radius: 16px; padding: 25px; border: 1px solid #34495e; color: #ecf0f1;">
        <div style="text-align: center; border-bottom: 1px solid #2c3e50; padding-bottom: 15px; margin-bottom: 20px;">
            <span style="background: {badge_bg}; color: {status_color}; font-size: 13px; font-weight: 800; padding: 6px 14px; border-radius: 20px; text-transform: uppercase; letter-spacing: 1px;">
                {status_title}
            </span>
            <h1 style="color: {status_color}; font-size: 48px; font-weight: 900; margin: 15px 0 5px 0;">{risk_pct}%</h1>
            <p style="color: #bdc3c7; font-size: 14px; margin: 0;">Predicted Cardiovascular Disease Risk Score</p>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #bdc3c7; margin-bottom: 6px;">
                <span>0% (Optimal)</span>
                <span>50% (Moderate)</span>
                <span>100% (High Risk)</span>
            </div>
            <div style="background: #2c3e50; height: 12px; border-radius: 6px; overflow: hidden;">
                <div style="background: {status_color}; width: {risk_pct}%; height: 100%; border-radius: 6px; transition: width 0.6s ease-in-out;"></div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr; gap: 12px; font-size: 14px;">
            <div style="background: #2c3e50; padding: 12px 16px; border-radius: 10px;">
                <strong style="color: #00cec9;">📋 Key Observations:</strong>
                <p style="margin: 4px 0 0 0; color: #dcdde1;">{risk_factors_str}</p>
            </div>
            <div style="background: #2c3e50; padding: 12px 16px; border-radius: 10px;">
                <strong style="color: #00cec9;">💡 Medical Guidance:</strong>
                <p style="margin: 4px 0 0 0; color: #dcdde1;">{recommendation}</p>
            </div>
        </div>
    </div>
    """

    return result_html

# ==========================================
# 3. Reset Function
# ==========================================
def reset_all_inputs():
    default_output_html = """
    <div style="background: #1e272e; border-radius: 16px; padding: 40px 20px; border: 1px solid #34495e; text-align: center; color: #95a5a6;">
        <p style="font-size: 16px; margin: 0;">Please select and enter patient parameters on the left, then click <b>RUN CLINICAL RISK ASSESSMENT</b>.</p>
    </div>
    """
    # Returns None for all inputs and resets output panel
    return (
        None, None, None, None, None, None, None,
        None, None, None,
        None, None, None,
        default_output_html
    )

# ==========================================
# 4. Custom CSS Theme
# ==========================================
dark_theme_css = """
.gradio-container {
    background-color: #0f172a !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

div.block, div.form {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
}

label, span.text-gray-500 {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

input[type="number"], select, .gr-input {
    background-color: #0f172a !important;
    border: 1px solid #475569 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}

button.primary-btn {
    background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
}

button.primary-btn:hover {
    background: linear-gradient(135deg, #0369a1 0%, #0f766e 100%) !important;
}

button.reset-btn {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

button.reset-btn:hover {
    background: #475569 !important;
}
"""

# ==========================================
# 5. Gradio Interface Layout
# ==========================================
with gr.Blocks() as app:

    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #0f172a, #1e293b); border: 1px solid #334155; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: #38bdf8; font-size: 32px; font-weight: 800; margin: 0; letter-spacing: -0.5px;">❤️ HEART DISEASE PREDICTION</h1>
        <p style="color: #94a3b8; font-size: 15px; margin-top: 6px;">Enter Patient Medical Details</p>
    </div>
    """)

    with gr.Row():
        # Left Panel - Input Controls
        with gr.Column(scale=3):
            with gr.Accordion("👤 Patient Baseline Vitals", open=True):
                with gr.Row():
                    age = gr.Number(value=None, label="Age (Years)", placeholder="Select / Enter Age...")
                    sex = gr.Radio(["Female", "Male"], value=None, label="Biological Sex")
                    fbs = gr.Radio(["No (<120 mg/dl)", "Yes (>120 mg/dl)"], value=None, label="Fasting Blood Sugar")

            with gr.Accordion("🩸 Hemodynamic & Stress Test Data", open=True):
                with gr.Row():
                    trestbps = gr.Number(value=None, label="Resting BP (mm Hg)", placeholder="Select / Enter BP...")
                    chol = gr.Number(value=None, label="Serum Cholesterol (mg/dl)", placeholder="Select / Enter Cholesterol...")
                with gr.Row():
                    thalach = gr.Number(value=None, label="Max Heart Rate (bpm)", placeholder="Select / Enter Max Heart Rate...")
                    exang = gr.Radio(["No", "Yes"], value=None, label="Exercise Induced Angina")

            with gr.Accordion("🔬 Advanced Diagnostics & Imaging", open=True):
                with gr.Row():
                    cp = gr.Dropdown(
                        choices=["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"],
                        value=None, label="Chest Pain Type"
                    )
                    restecg = gr.Dropdown(
                        choices=["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
                        value=None, label="Resting ECG Results"
                    )
                with gr.Row():
                    oldpeak = gr.Number(value=None, label="ST Depression (Oldpeak)", placeholder="Select / Enter Oldpeak...")
                    slope = gr.Dropdown(
                        choices=["0: Upsloping", "1: Flat", "2: Downsloping"],
                        value=None, label="Peak Exercise ST Slope"
                    )
                with gr.Row():
                    ca = gr.Dropdown(
                        choices=["0 Vessels", "1 Vessel", "2 Vessels", "3 Vessels", "4 Vessels"],
                        value=None, label="Fluoroscopy Major Vessels"
                    )
                    thal = gr.Dropdown(
                        choices=["0: Normal", "1: Fixed Defect", "2: Reversible Defect", "3: Unknown"],
                        value=None, label="Thalassemia Score"
                    )

            # Action Buttons Row
            with gr.Row():
                submit_btn = gr.Button("⚡ Risk Pridiction ", elem_classes=["primary-btn"], size="lg", scale=3)
                reset_btn = gr.Button("🔄 Reset ", elem_classes=["reset-btn"], size="lg", scale=1)

        # Right Panel - Medical Output Display
        with gr.Column(scale=2):
            gr.Markdown("<h3 style='color: #38bdf8; margin-top: 0;'>Diagnostic Analysis Output</h3>")
            output_display = gr.HTML("""
            <div style="background: #1e272e; border-radius: 16px; padding: 40px 20px; border: 1px solid #34495e; text-align: center; color: #95a5a6;">
                <p style="font-size: 16px; margin: 0;">Please select and enter patient parameters on the left, then click <b>RUN CLINICAL RISK ASSESSMENT</b>.</p>
            </div>
            """)

    all_input_components = [age, sex, fbs, trestbps, chol, thalach, exang, cp, restecg, oldpeak, slope, ca, thal]

    # Run Prediction Event
    submit_btn.click(
        fn=analyze_heart_risk,
        inputs=all_input_components,
        outputs=output_display
    )

    # Refresh / Reset Event
    reset_btn.click(
        fn=reset_all_inputs,
        inputs=None,
        outputs=all_input_components + [output_display]
    )

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), theme=gr.themes.Base(), css=dark_theme_css, share=False, debug=False)