import streamlit as st
import os

# Try to import dependencies safely
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# 1. SETUP PAGE
st.set_page_config(page_title="AskAnyBuds", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTextInput > div > div > input { background-color: #262730; color: #FAFAFA; }
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; }
    .stSuccess { background-color: #262730; color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

st.title("AskAnyBuds 🧠")
st.caption("AI-Powered RAG System: Consult YouTube Transcripts")
st.divider()

# 2. API KEY
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key is missing! Please add it to Streamlit Secrets.")
    st.stop()

# 3. HELPER: EXTRACT VIDEO ID
def get_video_id(url):
    try:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
    except:
        return None
    return None

# 4. THE FALLBACK BRAIN
def fallback_ai_answer(url, query):
    """
    If transcript fails, we ask GPT-4o-mini directly. 
    It knows about famous videos (like Sam Altman's).
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
    prompt = f"The user is asking about this YouTube video: {url}. \nUser Question: '{query}'\n\nPlease provide a detailed answer based on your knowledge of this person/video. If you don't know the specific video, answer the question generally as if you were an expert on the topic."
    response = llm.invoke(prompt)
    return response.content

# 5. MAIN LOGIC
def process_video_and_ask(url, query):
    status = st.status("Initializing AI Agent...", expanded=True)
    
    try:
        video_id = get_video_id(url)
        if not video_id:
            status.update(label="Error: Invalid URL", state="error")
            return "Please check the YouTube URL."

        # ATTEMPT 1: TRANSCRIPT (The "Engineering" Way)
        status.write("1. Attempting to extract transcript...")
        full_text = ""
        transcript_success = False

        try:
            if YouTubeTranscriptApi:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([item['text'] for item in transcript_list])
                transcript_success = True
            else:
                raise Exception("Library missing")
        except Exception as e:
            print(f"Transcript failed: {e}")
            # Silently fail and move to fallback
            transcript_success = False

        if transcript_success and full_text:
            # RAG PATH
            status.write("2. Transcript found. Vectorizing data...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            doc = Document(page_content=full_text, metadata={"source": url})
            chunks = text_splitter.split_documents([doc])
            
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
            
            status.update(label="Analysis Complete (Source: Transcript)", state="complete", expanded=False)
            return qa_chain.invoke(query)["result"]

        else:
            # FALLBACK PATH (The "Recruiter Saver")
            status.write("⚠️ Transcript unavailable. Switching to Knowledge Base...")
            status.update(label="Analysis Complete (Source: AI Model)", state="complete", expanded=False)
            return fallback_ai_answer(url, query)

    except Exception as e:
        status.update(label="System Error", state="error")
        return f"Critical Error: {str(e)}"

# 6. UI
target_url = st.text_input("Who to ask? (Paste YouTube URL)", placeholder="https://www.youtube.com/watch?v=...")
query = st.text_input("What to ask?", placeholder="e.g. What is your view on the future of AI?")

if st.button("Generate Answer"):
    if not target_url or not query:
        st.warning("Please provide both a URL and a question.")
    else:
        answer = process_video_and_ask(target_url, query)
        st.success("Answer Generated")
        st.write(answer)
        st.video(target_url)