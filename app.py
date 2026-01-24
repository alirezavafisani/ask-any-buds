import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. SETUP PAGE
st.set_page_config(page_title="AskAnyBuds", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTextInput > div > div > input { background-color: #262730; color: #FAFAFA; }
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("AskAnyBuds 🧠")
st.caption("AI-Powered RAG System: Consult YouTube Transcripts")
st.divider()

# 2. SIDEBAR - API KEY HANDLING
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key is missing! Please add it to Streamlit Secrets.")
    st.stop()

# 3. HELPER FUNCTION: EXTRACT VIDEO ID
def get_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    Examples:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# 4. HELPER FUNCTION: THE RAG PIPELINE
def process_video_and_ask(url, query):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return "Error: Could not extract Video ID. Check your URL."

        with st.status("Processing video...", expanded=True) as status:
            # A. DIRECT TRANSCRIPT FETCH (Bypassing LangChain Loader)
            st.write("1. Fetching raw transcript...")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine the list of dictionaries into a single string
            full_text = " ".join([item['text'] for item in transcript_list])
            
            # Manually create a LangChain Document
            documents = [Document(page_content=full_text, metadata={"source": url})]
            
            # B. SPLIT TEXT
            st.write("2. Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # C. EMBEDDING
            st.write("3. Creating vector embeddings...")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # D. VECTOR STORE
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # E. RETRIEVAL CHAIN
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            
            status.update(label="Brain is ready!", state="complete", expanded=False)
            
        # F. GET ANSWER
        response = qa_chain.invoke(query)
        return response["result"]

    except Exception as e:
        return f"Error: {str(e)}"

# 5. USER INTERFACE
target_url = st.text_input("Who to ask? (Paste YouTube URL)", placeholder="https://www.youtube.com/watch?v=...")
query = st.text_input("What to ask?", placeholder="e.g. What is your view on the future of AI?")

if st.button("Generate Answer"):
    if not target_url or not query:
        st.warning("Please provide both a URL and a question.")
    else:
        answer = process_video_and_ask(target_url, query)
        st.success("Analysis Complete")
        st.markdown(f"### Answer:\n{answer}")
        st.video(target_url)