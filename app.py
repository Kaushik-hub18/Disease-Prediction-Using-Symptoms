# app.py  –  NO chatbot code, polished UI + export buttons
import streamlit as st, pandas as pd, numpy as np, joblib, json, re, io, datetime
from streamlit_tags import st_tags
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- utilities ----------
def std_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()

# ---------- cached loaders ----------
@st.cache_resource
def load_model():          return joblib.load("model.pkl")

@st.cache_data
def load_symptom_list():   return list(pd.read_csv("symptoms_clean.csv").drop(columns="Disease").columns)

@st.cache_data
def load_disease_info():
    with open("disease_info.json", encoding="utf‑8") as f:
        raw = json.load(f)
    return {std_name(k): v for k, v in raw.items()}

model, symptoms_list, disease_info = load_model(), load_symptom_list(), load_disease_info()

# ---------- page/header ----------
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")
col_l, col_r = st.columns([1, 6])
with col_l:
    try: st.image("logo.png", width=90)
    except FileNotFoundError: st.write("🩺")
with col_r:
    st.session_state.dark = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark", False))
    if st.session_state.dark:
        st.markdown("<style>.stApp{background:#111;color:#eee}.stTags{background:#222!important}</style>", unsafe_allow_html=True)

st.title("🩺 Disease Predictor from Symptoms")

# ---------- symptom entry ----------
st.subheader("Enter your symptoms")
selected = st.multiselect(
    "Start typing to search (you can pick many):",
    options=symptoms_list,
    max_selections=10
)


# ---------- prediction ----------
if st.button("Predict"):
    if not selected:
        st.warning("Please add at least one symptom."); st.stop()

    X = pd.DataFrame([np.zeros(len(symptoms_list))], columns=symptoms_list)
    X.loc[0, selected] = 1
    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[-3:][::-1]

    st.markdown("---"); st.header("🔎 Results")
    report_rows = []

    for dis, pr in zip(model.classes_[top_idx], probs[top_idx]):
        info = disease_info.get(std_name(dis), {})
        desc = info.get("description", "No description available.")
        rem  = info.get("remedy", "Consult a qualified medical professional.")
        st.subheader(f"{dis} – {pr*100:.1f}%"); st.write(desc); st.info(rem)

        report_rows.append({"Disease": dis, "Probability (%)": f"{pr*100:.1f}", "Description": desc, "Remedy": rem})

    if report_rows:  # export buttons
        df = pd.DataFrame(report_rows)
        csv_bytes = df.to_csv(index=False).encode("utf‑8")

        pdf = io.BytesIO(); c = canvas.Canvas(pdf, pagesize=A4); w, h = A4; y = h - 50
        c.setFont("Helvetica-Bold", 14); c.drawString(40, y, "Disease Prediction Report"); y -= 30
        c.setFont("Helvetica", 11); c.drawString(40, y, "Generated: "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")); y -= 40

        for r in report_rows:
            c.setFont("Helvetica-Bold", 12); c.drawString(40, y, f"{r['Disease']} ({r['Probability (%)']}%)"); y -= 18
            c.setFont("Helvetica", 10)
            for txt in [r["Description"], "Remedy: "+r["Remedy"]]:
                for chunk in [txt[i:i+100] for i in range(0, len(txt), 100)]:
                    c.drawString(50, y, chunk); y -= 14
            y -= 14
            if y < 60: c.showPage(); y = h - 50
        c.save(); pdf.seek(0)

        st.download_button("⬇️ Download CSV", csv_bytes, "disease_report.csv", "text/csv")
        st.download_button("⬇️ Download PDF", pdf, "disease_report.pdf", "application/pdf")

st.markdown("---")
st.caption("⚠️ Educational tool – not a substitute for professional medical advice.")
