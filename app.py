import json
import os
from datetime import datetime

import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Patient Health Analyzer",
    page_icon="🩺",
    layout="wide",
)

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

SYSTEM_PROMPT = """You are a clinical decision-SUPPORT assistant for hospital clinicians.
You analyze structured patient information and return a careful, evidence-informed
assessment. You do NOT make final diagnoses and you do NOT replace clinical judgment.

Rules:
- Base your analysis only on the data provided. If key data is missing, say so.
- Express diagnoses as "possible conditions" with likelihood (Low/Moderate/High) and reasoning.
- Always highlight red flags that need urgent clinician attention.
- Consider allergies and current medications before suggesting any treatment strategy.
- Do not give exact drug doses. Suggest drug classes/options for clinician review only.
- Be concise, specific and clinically sensible.

Return ONLY valid JSON with exactly this structure:
{
  "summary": "2-4 sentence clinical summary",
  "overall_risk_level": "Low | Moderate | High | Critical",
  "health_risks": [{"risk": "", "level": "Low|Moderate|High", "rationale": ""}],
  "possible_conditions": [{"condition": "", "likelihood": "Low|Moderate|High", "reasoning": ""}],
  "recommended_tests": ["..."],
  "treatment_support": [{"strategy": "", "details": ""}],
  "medication_considerations": ["..."],
  "lifestyle_recommendations": ["..."],
  "red_flags": ["..."],
  "follow_up_plan": "",
  "data_gaps": ["information that would improve this analysis"],
  "case_specific_limitations": ["limitations of AI analysis for this specific case"]
}"""

DISCLAIMER = (
    "⚠️ **Clinical decision support only.** Output is AI-generated, may be incomplete or "
    "wrong, and must be reviewed by a qualified clinician. Not a diagnosis. Do not enter "
    "real patient identifiers unless your hospital's privacy policy and Groq data terms permit it."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    """Priority: sidebar input -> Streamlit secrets -> environment variable."""
    typed = st.session_state.get("api_key_input", "").strip()
    if typed:
        return typed
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def calc_bmi(weight_kg: float, height_cm: float):
    if weight_kg and height_cm:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return round(bmi, 1), cat
    return None, "N/A"


def vital_flags(p: dict) -> list:
    """Simple rule-based screening on vitals (runs locally, independent of the LLM)."""
    flags = []
    if p["systolic"] >= 180 or p["diastolic"] >= 120:
        flags.append(("🔴", "Hypertensive crisis range BP"))
    elif p["systolic"] >= 140 or p["diastolic"] >= 90:
        flags.append(("🟠", "Elevated BP (hypertension range)"))
    elif p["systolic"] < 90 or p["diastolic"] < 60:
        flags.append(("🟠", "Low BP (hypotension range)"))
    if p["heart_rate"] > 100:
        flags.append(("🟠", "Tachycardia (HR > 100)"))
    elif p["heart_rate"] < 60:
        flags.append(("🟡", "Bradycardia (HR < 60)"))
    if p["spo2"] < 90:
        flags.append(("🔴", "Severe hypoxemia (SpO₂ < 90%)"))
    elif p["spo2"] < 94:
        flags.append(("🟠", "Low oxygen saturation (SpO₂ < 94%)"))
    if p["temp_c"] >= 38.0:
        flags.append(("🟠", "Fever (≥ 38.0 °C)"))
    elif p["temp_c"] < 35.0:
        flags.append(("🟠", "Hypothermia (< 35.0 °C)"))
    if p["resp_rate"] > 20:
        flags.append(("🟠", "Tachypnea (RR > 20)"))
    if p["glucose"] and p["glucose"] >= 200:
        flags.append(("🟠", "High random glucose (≥ 200 mg/dL)"))
    elif p["glucose"] and p["glucose"] < 70:
        flags.append(("🟠", "Low glucose (< 70 mg/dL)"))
    return flags


def build_patient_text(p: dict) -> str:
    bmi, bmi_cat = calc_bmi(p["weight"], p["height"])
    return f"""PATIENT DATA
Age: {p['age']} | Sex: {p['sex']}
Height: {p['height']} cm | Weight: {p['weight']} kg | BMI: {bmi} ({bmi_cat})

VITALS
BP: {p['systolic']}/{p['diastolic']} mmHg | Heart rate: {p['heart_rate']} bpm
Temp: {p['temp_c']} C | SpO2: {p['spo2']}% | Resp rate: {p['resp_rate']}/min
Random glucose: {p['glucose'] if p['glucose'] else 'not provided'} mg/dL

PRESENTING SYMPTOMS: {p['symptoms'] or 'not provided'}
SYMPTOM DURATION: {p['duration'] or 'not provided'}
MEDICAL HISTORY: {p['history'] or 'none provided'}
FAMILY HISTORY: {p['family_history'] or 'none provided'}
CURRENT MEDICATIONS: {p['medications'] or 'none provided'}
ALLERGIES: {p['allergies'] or 'none provided'}
LIFESTYLE: Smoking: {p['smoking']} | Alcohol: {p['alcohol']} | Activity: {p['activity']}
LAB RESULTS / OTHER NOTES: {p['labs'] or 'none provided'}"""


def run_analysis(client: Groq, model: str, patient_text: str, temperature: float) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_text},
        ],
        temperature=temperature,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def report_to_markdown(p: dict, r: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) if items else "- None"

    lines = [
        "# AI Patient Health Analysis Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "> Clinical decision support only. Must be reviewed by a qualified clinician.",
        "",
        f"**Patient:** {p['name'] or 'N/A'} | **Age/Sex:** {p['age']} / {p['sex']}",
        "",
        f"## Summary\n{r.get('summary', '')}",
        f"\n**Overall risk level:** {r.get('overall_risk_level', 'N/A')}",
        "\n## Health Risks",
    ]
    for x in r.get("health_risks", []):
        lines.append(f"- **{x.get('risk')}** ({x.get('level')}): {x.get('rationale')}")
    lines.append("\n## Possible Conditions")
    for x in r.get("possible_conditions", []):
        lines.append(f"- **{x.get('condition')}** ({x.get('likelihood')}): {x.get('reasoning')}")
    lines.append("\n## Recommended Tests\n" + bullets(r.get("recommended_tests")))
    lines.append("\n## Treatment Support Strategies")
    for x in r.get("treatment_support", []):
        lines.append(f"- **{x.get('strategy')}**: {x.get('details')}")
    lines.append("\n## Medication Considerations\n" + bullets(r.get("medication_considerations")))
    lines.append("\n## Lifestyle Recommendations\n" + bullets(r.get("lifestyle_recommendations")))
    lines.append("\n## Red Flags\n" + bullets(r.get("red_flags")))
    lines.append(f"\n## Follow-up Plan\n{r.get('follow_up_plan', '')}")
    lines.append("\n## Data Gaps\n" + bullets(r.get("data_gaps")))
    lines.append("\n## Case-specific Limitations\n" + bullets(r.get("case_specific_limitations")))
    return "\n".join(lines)


RISK_COLORS = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.text_input(
        "Groq API key",
        type="password",
        key="api_key_input",
        help="Or set GROQ_API_KEY as an environment variable / in .streamlit/secrets.toml",
    )
    model = st.selectbox("Model", MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Lower = more consistent, clinical-style output")
    st.divider()
    st.caption(DISCLAIMER)

# ----------------------------------------------------------------------------
# Header + tabs
# ----------------------------------------------------------------------------
st.title("🩺 AI Patient Health Analyzer")
st.caption("AI-assisted risk identification and treatment-support suggestions for clinicians")

tab_input, tab_results, tab_chat, tab_eval = st.tabs(
    ["📝 Patient Input", "📊 Analysis", "💬 Ask Follow-ups", "⚖️ Benefits & Limitations"]
)

# ----------------------------------------------------------------------------
# Tab 1: Input
# ----------------------------------------------------------------------------
with tab_input:
    with st.form("patient_form"):
        st.subheader("Demographics")
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Patient name / ID (optional)")
        age = c2.number_input("Age", 0, 120, 45)
        sex = c3.selectbox("Sex", ["Male", "Female", "Other"])
        height = c4.number_input("Height (cm)", 30.0, 250.0, 170.0)
        weight = st.number_input("Weight (kg)", 2.0, 300.0, 75.0)

        st.subheader("Vitals")
        v1, v2, v3, v4 = st.columns(4)
        systolic = v1.number_input("Systolic BP", 50, 260, 120)
        diastolic = v2.number_input("Diastolic BP", 30, 160, 80)
        heart_rate = v3.number_input("Heart rate (bpm)", 20, 250, 78)
        resp_rate = v4.number_input("Resp. rate (/min)", 5, 60, 16)
        v5, v6, v7 = st.columns(3)
        temp_c = v5.number_input("Temperature (°C)", 30.0, 43.0, 36.8, step=0.1)
        spo2 = v6.number_input("SpO₂ (%)", 50, 100, 98)
        glucose = v7.number_input("Random glucose (mg/dL, 0 = unknown)", 0, 800, 0)

        st.subheader("Clinical Information")
        symptoms = st.text_area("Presenting symptoms", placeholder="e.g. chest tightness on exertion, fatigue, shortness of breath")
        duration = st.text_input("Symptom duration", placeholder="e.g. 2 weeks, worsening")
        history = st.text_area("Past medical history", placeholder="e.g. type 2 diabetes, hypertension, prior MI")
        family_history = st.text_area("Family history", placeholder="e.g. father had heart disease at 55")
        medications = st.text_area("Current medications", placeholder="e.g. metformin, amlodipine")
        allergies = st.text_input("Allergies", placeholder="e.g. penicillin")

        l1, l2, l3 = st.columns(3)
        smoking = l1.selectbox("Smoking", ["Never", "Former", "Current"])
        alcohol = l2.selectbox("Alcohol", ["None", "Occasional", "Moderate", "Heavy"])
        activity = l3.selectbox("Physical activity", ["Sedentary", "Light", "Moderate", "Active"])

        labs = st.text_area("Lab results / other notes", placeholder="e.g. HbA1c 8.2%, LDL 165 mg/dL, creatinine 1.4")

        submitted = st.form_submit_button("🔍 Analyze Patient", type="primary", use_container_width=True)

    if submitted:
        patient = dict(
            name=name, age=age, sex=sex, height=height, weight=weight,
            systolic=systolic, diastolic=diastolic, heart_rate=heart_rate,
            resp_rate=resp_rate, temp_c=temp_c, spo2=spo2, glucose=glucose,
            symptoms=symptoms, duration=duration, history=history,
            family_history=family_history, medications=medications,
            allergies=allergies, smoking=smoking, alcohol=alcohol,
            activity=activity, labs=labs,
        )
        api_key = get_api_key()
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not symptoms.strip() and not history.strip():
            st.warning("Please enter at least symptoms or medical history.")
        else:
            try:
                with st.spinner("Analyzing patient data..."):
                    client = Groq(api_key=api_key)
                    patient_text = build_patient_text(patient)
                    result = run_analysis(client, model, patient_text, temperature)
                st.session_state["patient"] = patient
                st.session_state["patient_text"] = patient_text
                st.session_state["result"] = result
                st.session_state["chat"] = []
                st.success("Analysis complete — open the **📊 Analysis** tab.")
            except json.JSONDecodeError:
                st.error("The model returned malformed output. Please try again.")
            except Exception as e:
                st.error(f"Error calling Groq API: {e}")

# ----------------------------------------------------------------------------
# Tab 2: Results
# ----------------------------------------------------------------------------
with tab_results:
    if "result" not in st.session_state:
        st.info("Submit patient data in the **Patient Input** tab to see the analysis.")
    else:
        p = st.session_state["patient"]
        r = st.session_state["result"]

        bmi, bmi_cat = calc_bmi(p["weight"], p["height"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMI", bmi, bmi_cat)
        m2.metric("Blood pressure", f"{p['systolic']}/{p['diastolic']}")
        m3.metric("SpO₂", f"{p['spo2']}%")
        level = r.get("overall_risk_level", "N/A")
        m4.metric("AI overall risk", f"{RISK_COLORS.get(level, '')} {level}")

        flags = vital_flags(p)
        if flags:
            st.markdown("#### Automated vital-sign screening")
            for icon, text in flags:
                st.markdown(f"{icon} {text}")

        if r.get("red_flags"):
            st.error("**🚩 Red flags needing clinician attention**\n\n" +
                     "\n".join(f"- {x}" for x in r["red_flags"]))

        st.markdown("### Summary")
        st.write(r.get("summary", ""))

        st.markdown("### Health Risks")
        for x in r.get("health_risks", []):
            st.markdown(f"{RISK_COLORS.get(x.get('level'), '•')} **{x.get('risk')}** "
                        f"({x.get('level')}) — {x.get('rationale')}")

        st.markdown("### Possible Conditions (for clinician review)")
        for x in r.get("possible_conditions", []):
            with st.expander(f"{x.get('condition')} — likelihood: {x.get('likelihood')}"):
                st.write(x.get("reasoning", ""))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Recommended Tests")
            for t in r.get("recommended_tests", []):
                st.markdown(f"- {t}")
            st.markdown("### Medication Considerations")
            for t in r.get("medication_considerations", []):
                st.markdown(f"- {t}")
        with col_b:
            st.markdown("### Treatment Support Strategies")
            for x in r.get("treatment_support", []):
                st.markdown(f"- **{x.get('strategy')}**: {x.get('details')}")
            st.markdown("### Lifestyle Recommendations")
            for t in r.get("lifestyle_recommendations", []):
                st.markdown(f"- {t}")

        st.markdown("### Follow-up Plan")
        st.write(r.get("follow_up_plan", ""))

        with st.expander("Data gaps & limitations for this case"):
            st.markdown("**Missing information that would improve the analysis**")
            for t in r.get("data_gaps", []):
                st.markdown(f"- {t}")
            st.markdown("**Case-specific AI limitations**")
            for t in r.get("case_specific_limitations", []):
                st.markdown(f"- {t}")

        st.divider()
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=report_to_markdown(p, r),
            file_name=f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )
        st.caption(DISCLAIMER)

# ----------------------------------------------------------------------------
# Tab 3: Follow-up chat
# ----------------------------------------------------------------------------
with tab_chat:
    if "result" not in st.session_state:
        st.info("Run an analysis first, then ask follow-up questions about the case here.")
    else:
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask about this case (e.g. 'What would change if the patient is pregnant?')")
        if question:
            st.session_state["chat"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            api_key = get_api_key()
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar.")
            else:
                context = (
                    "You are a clinical decision-support assistant helping a clinician "
                    "discuss a case. Be concise, flag uncertainty, never give final diagnoses "
                    "or exact drug doses, and remind that a clinician must make final decisions.\n\n"
                    f"{st.session_state['patient_text']}\n\n"
                    f"PRIOR AI ANALYSIS:\n{json.dumps(st.session_state['result'])}"
                )
                messages = [{"role": "system", "content": context}] + st.session_state["chat"]
                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            client = Groq(api_key=api_key)
                            resp = client.chat.completions.create(
                                model=model, messages=messages,
                                temperature=temperature, max_tokens=1200,
                            )
                            answer = resp.choices[0].message.content
                        st.markdown(answer)
                    st.session_state["chat"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error calling Groq API: {e}")

# ----------------------------------------------------------------------------
# Tab 4: Benefits & limitations evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation of AI-Assisted Healthcare Solutions")
    b, l = st.columns(2)
    with b:
        st.markdown("### ✅ Benefits")
        st.markdown(
            """
- **Speed:** summarizes complex histories and surfaces risks in seconds.
- **Consistency:** applies the same structured review to every patient, reducing variation.
- **Safety net:** can flag red flags, drug–allergy conflicts, or missed risk factors.
- **Workload relief:** reduces documentation and information-synthesis burden on clinicians.
- **Access:** supports clinicians in resource-limited or high-volume settings.
- **Education:** gives transparent reasoning that supports learning and second opinions.
- **Preventive care:** helps spot early risk patterns (e.g. cardiometabolic risk) for lifestyle intervention.
"""
        )
    with l:
        st.markdown("### ⚠️ Limitations & Risks")
        st.markdown(
            """
- **Hallucination:** LLMs can state incorrect facts or invent plausible-sounding reasoning.
- **No physical exam or imaging:** cannot assess what isn't in the text input.
- **Data quality dependence:** incomplete or wrong inputs produce wrong outputs.
- **Bias:** training data may under-represent certain populations, skewing recommendations.
- **Automation bias:** clinicians may over-trust AI suggestions.
- **Privacy & compliance:** patient data handling (HIPAA/GDPR/local law) must be addressed.
- **Accountability & regulation:** unclear liability; clinical-grade tools typically need validation and regulatory approval.
- **Not up to date:** model knowledge may lag current guidelines.
- **Lack of empathy:** cannot replace patient communication and shared decision-making.
"""
        )

    st.markdown("### 🧭 Recommended Safeguards")
    st.markdown(
        """
1. Keep a **human-in-the-loop**: a licensed clinician reviews every output.
2. **De-identify** patient data and use compliant, approved infrastructure.
3. **Validate** against local guidelines and audit performance regularly.
4. Display **uncertainty and data gaps** clearly (as this app does).
5. Log usage for **auditing and accountability**; train staff on appropriate use.
"""
    )
    st.caption(DISCLAIMER)
