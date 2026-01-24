import streamlit as st
import os
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import yt_dlp
from pydub import AudioSegment

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AskAnyBuds",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333;
        border-radius: 8px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #fff;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stButton > button:hover {
        border-color: #4a9eff;
        box-shadow: 0 0 20px rgba(74, 158, 255, 0.3);
    }
    h1 { text-align: center; font-weight: 300; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# AskAnyBuds")
st.markdown("<p style='text-align:center;color:#666;'>Hear answers in their own voice</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================
# API KEY
# ============================================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return None


def get_transcript_with_timestamps(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    processed = []
    for item in transcript_list:
        processed.append({
            "text": item["text"],
            "start": item["start"],
            "end": item["start"] + item["duration"]
        })
    return processed


def create_documents_with_timestamps(transcript_segments, chunk_size=500):
    documents = []
    current_text = ""
    current_start = None
    current_end = None
    
    for seg in transcript_segments:
        if current_start is None:
            current_start = seg["start"]
        
        current_text += " " + seg["text"]
        current_end = seg["end"]
        
        if len(current_text) >= chunk_size:
            doc = Document(
                page_content=current_text.strip(),
                metadata={"start": current_start, "end": current_end}
            )
            documents.append(doc)
            current_text = ""
            current_start = None
            current_end = None
    
    if current_text.strip():
        doc = Document(
            page_content=current_text.strip(),
            metadata={"start": current_start, "end": current_end}
        )
        documents.append(doc)
    
    return documents


def semantic_search(documents, query, api_key, top_k=3):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    processed_results = []
    for doc, distance in results:
        similarity = max(0, min(100, (1 - distance / 2) * 100))
        processed_results.append((doc, similarity))
    
    return processed_results


def download_audio(video_url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    return output_path + ".mp3"


def cut_and_concatenate_audio(audio_path, segments, output_path):
    full_audio = AudioSegment.from_mp3(audio_path)
    combined = AudioSegment.empty()
    
    for start, end in segments:
        start_ms = max(0, int(start * 1000) - 300)
        end_ms = min(len(full_audio), int(end * 1000) + 300)
        clip = full_audio[start_ms:end_ms]
        clip = clip.fade_in(100).fade_out(100)
        combined += clip + AudioSegment.silent(duration=200)
    
    combined.export(output_path, format="mp3")
    return output_path


# ============================================
# MAIN UI
# ============================================

who = st.text_input("Who to ask?", placeholder="Paste YouTube URL here")
what = st.text_input("What to ask?", placeholder="Your question")

if st.button("🎙️ Generate Answer"):
    if not who or not what:
        st.warning("Please fill in both fields.")
        st.stop()
    
    video_id = extract_video_id(who)
    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()
    
    status = st.status("Processing...", expanded=True)
    
    try:
        status.write("📝 Fetching transcript...")
        transcript = get_transcript_with_timestamps(video_id)
        status.write(f"   Found {len(transcript)} segments")
        
        status.write("📦 Creating chunks...")
        documents = create_documents_with_timestamps(transcript)
        status.write(f"   Created {len(documents)} chunks")
        
        status.write("🔍 Searching for relevant segments...")
        results = semantic_search(documents, what, api_key, top_k=3)
        avg_score = sum(score for _, score in results) / len(results)
        status.write(f"   Relevance: {avg_score:.1f}%")
        
        status.write("⬇️ Downloading audio...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_base = os.path.join(tmpdir, "audio")
            audio_path = download_audio(who, audio_base)
            status.write("   Download complete")
            
            segments = [(doc.metadata["start"], doc.metadata["end"]) for doc, _ in results]
            segments.sort(key=lambda x: x[0])
            
            status.write("✂️ Cutting and joining clips...")
            output_path = os.path.join(tmpdir, "answer.mp3")
            cut_and_concatenate_audio(audio_path, segments, output_path)
            
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
        
        status.update(label="✅ Done!", state="complete", expanded=False)
        
        st.markdown("---")
        st.markdown("### 🎧 Their Answer")
        st.audio(audio_bytes, format="audio/mp3")
        
        st.markdown(f"**Relevance Score:** {avg_score:.1f}%")
        
        with st.expander("📜 Transcript"):
            for doc, score in results:
                st.write(doc.page_content)
                st.caption(f"Timestamp: {doc.metadata['start']:.1f}s to {doc.metadata['end']:.1f}s")
    
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"Error: {str(e)}")
        st.info("Make sure the video has captions enabled.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>RAG + Audio Processing Pipeline</p>", unsafe_allow_html=True)