import streamlit as st
import pdfplumber
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# --- 1. CONFIGURATION & MODEL LOADING ---
st.set_page_config(
    page_title=" keyword Extractor", 
    page_icon="🎓", 
    layout="wide"
)

# Custom CSS for the "Skill Chips" and UI styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .skill-chip {
        display: inline-block;
        padding: 8px 18px;
        margin: 6px;
        border-radius: 25px;
        background-color: #1E3A8A; /* Deep Blue */
        color: white;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid #1e40af;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1E3A8A;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


import os

@st.cache_resource
def load_ner_pipeline():
    """Loads the BERT model and tokenizer using an absolute local path."""
    # 1. Get the absolute path to your folder
    # This avoids the "Repo id" naming restriction error
    base_path = os.path.abspath("./my_ner_model/final_ner_model") 
    
    # 2. Load with local_files_only=True
    model = AutoModelForTokenClassification.from_pretrained(
        base_path, 
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, 
        local_files_only=True
    )
    
    return pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )

def extract_skills(text, nlp_pipe, threshold=0.90):
    """Processes text through BERT and returns a cleaned list of skills."""
    results = nlp_pipe(text)
    # Filter by confidence and specific 'SKILL' entity group
    skills = [entity['word'].strip() for entity in results if entity['score'] > threshold]
    # Remove duplicates and empty strings
    return sorted(list(set([s for s in skills if s])))

# --- 2. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.info("Adjust the sensitivity of the BERT Classification Head.")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.90, help="Lower values extract more keywords but may include noise.")
    st.divider()
    st.markdown("### System Details")
    st.write("**Model:** BERT-Base-Uncased")
    st.write("**Task:** Token Classification (NER)")
    st.write("**Architecture:** Encoder + Linear Head")

# --- 3. MAIN UI LAYOUT ---
st.title("Automated Keyword Extraction from Job Descriptions")
st.write("Upload a Job Description (JD) to automatically extract important keywords  using a fine-tuned NLP model.")

uploaded_file = st.file_uploader("📤 Upload Job Description (PDF)", type="pdf")

if uploaded_file:
    with st.spinner('🧠 BERT is analyzing the context...'):
        try:
            # Step 1: PDF Text Extraction
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            
            if not full_text.strip():
                st.error("❌ Error: Could not read text from this PDF. Please ensure it's not a scanned image.")
            else:
                # Step 2: Inference
                nlp_ner = load_ner_pipeline()
                extracted_keywords = extract_skills(full_text, nlp_ner, threshold=conf_threshold)

                # Step 3: Visualization
                st.divider()
                
                # Layout for Results
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("### 🎯 Identified keywords")
                    if extracted_keywords:
                        # Render modern chips
                        skill_html = "".join([f'<div class="skill-chip">{skill}</div>' for skill in extracted_keywords])
                        st.markdown(skill_html, unsafe_allow_html=True)
                        
                        # Export Options
                        st.write("")
                        df = pd.DataFrame(extracted_keywords, columns=["Keyword"])
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Keywords as CSV",
                            data=csv,
                            file_name='extracted_keywords.csv',
                            mime='text/csv',
                        )
                    else:
                        st.warning("No keywords detected above the threshold. Try lowering the 'Confidence Threshold' in the sidebar.")

                with col_right:
                    st.markdown("### 📄 Job Description Preview")
                    # Display raw text in a clean scrollable box
                    st.text_area("Source Text (First 2000 chars)", full_text[:2000], height=350)

                # Step 4: Final Success Note
                st.toast(f"Successfully extracted {len(extracted_keywords)} keywords!", icon='✅')

        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {str(e)}")

# --- 4. FOOTER ---
st.divider()
