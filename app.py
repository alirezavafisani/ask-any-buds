import streamlit as st

# 1. PAGE CONFIGURATION
# This sets the tab title and the dark mode icon
st.set_page_config(page_title="AskAnyBuds", page_icon="🧠", layout="centered")

# 2. CSS STYLING (THE DARK THEME)
# Streamlit has a default dark mode, but we enforce specific "serious" vibes here.
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# 3. THE HEADER
st.title("AskAnyBuds 🧠")
st.markdown("### Consult the minds of public figures.")
st.divider()

# 4. THE INPUTS (The UI you requested)
# We use columns to make it look cleaner, or standard inputs.
# Let's stick to a clean vertical layout for mobile friendliness.

target_url = st.text_input("1. Who to ask? (Paste YouTube URL)", placeholder="e.g., https://www.youtube.com/watch?v=...")
query = st.text_input("2. What to ask?", placeholder="e.g., What is the meaning of life?")

# 5. THE BUTTON
if st.button("Generate Answer"):
    if not target_url or not query:
        st.warning("Please provide both a URL and a question.")
    else:
        st.info("System is ready. Brain logic pending...")