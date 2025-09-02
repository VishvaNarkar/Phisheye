# app.py
import streamlit as st
import pandas as pd
from utils import parse_eml_bytes, basic_clean
from heuristics import run_heuristics
from model import predict_proba

st.set_page_config(page_title="PhishEye", page_icon="🐟", layout="wide")

st.title("PhishEye 🐟 — AI Email Security Assistant")
st.caption("Paste email text or upload a .eml / .txt file. The model + heuristics will assess phishing risk.")

tab1, tab2 = st.tabs(["Paste Text", "Upload File"])

email_text = ""

with tab1:
    email_text = st.text_area("Email content", height=250, placeholder="Paste email body here...")
with tab2:
    up = st.file_uploader("Upload .eml or .txt", type=["eml","txt"])
    if up:
        raw = up.read()
        if up.name.lower().endswith(".eml"):
            email_text = parse_eml_bytes(raw)
        else:
            email_text = raw.decode("utf-8","ignore")

colL, colR = st.columns([2,1])

with colL:
    st.subheader("Preview")
    st.write(email_text[:3000] if email_text else "_No content_")

if st.button("Analyze", type="primary") and email_text.strip():
    # ML score
    ml_score = predict_proba(email_text)  # 0..1 phishing prob
    # Heuristics
    findings, heur_score = run_heuristics(email_text)

    # simple fusion
    fused = min(1.0, 0.8*ml_score + 0.2*(ml_score + heur_score))

    risk = (
        "High Risk" if fused >= 0.75 else
        "Medium Risk" if fused >= 0.45 else
        "Low Risk"
    )

    with colR:
        st.metric("ML Phishing Probability", f"{ml_score:.2%}")
        st.metric("Heuristics Score (aux)", f"{heur_score:.2%}")
        st.metric("Final Risk", risk)

    st.divider()
    st.subheader("Heuristic Findings")

    st.write("**Content red flags:**")
    if findings["content_flags"]:
        st.write("- " + "\n- ".join(findings["content_flags"]))
    else:
        st.write("_None_")

    st.write("**URL analysis:**")
    url_rows = []
    for u in findings["urls"]:
        url_rows.append({"URL": u["url"], "Flags": ", ".join(u["flags"]) if u["flags"] else "—"})
    if url_rows:
        st.dataframe(pd.DataFrame(url_rows))
    else:
        st.write("_No URLs detected_")

    st.info("Note: This demo assists users but does not replace enterprise email security.")
