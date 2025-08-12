import streamlit as st, os, sys, joblib
st.title("NAFLD – smoke test")
try:
    import numpy as np, sklearn
    st.write("Python:", sys.version.split()[0])
    st.write("NumPy:", np.__version__, "sklearn:", sklearn.__version__)
except Exception as e:
    st.error(f"Import error: {e}")
st.write("Repo root files:", os.listdir("."))
st.write("Has nafld_pipeline.pkl?", os.path.exists("nafld_pipeline.pkl"))
if os.path.exists("nafld_pipeline.pkl"):
    try:
        pipe = joblib.load("nafld_pipeline.pkl")
        st.success(f"Model loaded: {type(pipe).__name__}")
    except Exception as e:
        st.error(f"Failed to load model: {type(e).__name__}: {e}")
