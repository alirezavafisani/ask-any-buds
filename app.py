import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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
# This allows it to work both locally (if you have .env) and on Streamlit Cloud (secrets)
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API Key is missing! Please add it to Streamlit Secrets.")
    st.stop()

# 3. HELPER FUNCTION: THE RAG PIPELINE
# This function is the "Engine" of your application.
def process_video_and_ask(url, query):
    try:
        # A. LOAD DATA
        with st.status("Processing video...", expanded=True) as status:
            st.write("1. Extracting transcript from YouTube...")
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            documents = loader.load()
            
            # B. SPLIT TEXT
            # We chop the text into chunks of 1000 characters so the AI can digest it.
            st.write("2. Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # C. EMBEDDING (The "Math" part)
            # We turn text into vectors using OpenAI
            st.write("3. Creating vector embeddings...")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # D. VECTOR STORE
            # We store vectors in FAISS (Facebook AI Similarity Search)
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # E. RETRIEVAL CHAIN
            # We set up the LLM (GPT-4o-mini is fast and cheap)
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", # "Stuff" puts all relevant chunks into the prompt
                retriever=vector_store.as_retriever()
            )
            
            status.update(label="Brain is ready!", state="complete", expanded=False)
            
        # F. GET ANSWER
        response = qa_chain.invoke(query)
        return response["result"]

    except Exception as e:
        return f"Error: {str(e)}"

# 4. USER INTERFACE
target_url = st.text_input("1. Who to ask? (Paste YouTube URL)", placeholder="https://www.youtube.com/watch?v=...")
query = st.text_input("2. What to ask?", placeholder="e.g. What is your view on the future of AI?")

if st.button("Generate Answer"):
    if not target_url or not query:
        st.warning("Please provide both a URL and a question.")
    else:
        # Run the AI
        answer = process_video_and_ask(target_url, query)
        
        # Display Result
        st.success("Analysis Complete")
        st.markdown(f"### Answer:\n{answer}")
        
        # Display the video for reference
        st.video(target_url)