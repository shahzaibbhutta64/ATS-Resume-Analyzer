# ATS Resume Scorer

Upload a resume (PDF/DOCX/Image) and get an ATS compatibility score plus
section-by-section, ATS-friendly improvement suggestions — powered by Gemini.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deployment
Deployed on Streamlit Community Cloud. Add your Gemini API key as a secret
named GEMINI_API_KEY, or paste it into the sidebar at runtime.
