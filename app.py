import streamlit as st, sys
st.title("Hello from a clean app")
try:
    import numpy as np, sklearn
    st.caption(f"Python {sys.version.split()[0]} • numpy {getattr(np,'__version__','(not installed)')} • sklearn {getattr(sklearn,'__version__','(not installed)')}")
except Exception as e:
    st.error(f"Import error: {e}")
