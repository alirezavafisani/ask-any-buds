import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import scrapetube

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AskAnyBuds",
    page_icon="A",
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
st.markdown("<p style='text-align:center;color:#666;'>Ask any YouTuber anything. Search their entire channel.</p>", unsafe_allow_html=True)
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

def extract_channel_id(url):
    if "/@" in url:
        return url.split("/@")[1].split("/")[0].split("?")[0]
    elif "/channel/" in url:
        return url.split("/channel/")[1].split("/")[0].split("?")[0]
    elif "/c/" in url:
        return url.split("/c/")[1].split("/")[0].split("?")[0]
    return url.strip()


def get_channel_videos(channel_handle, limit=50):
    videos = []
    try:
        for video in scrapetube.get_channel(channel_username=channel_handle, limit=limit):
            videos.append({
                "id": video["videoId"],
                "title": video.get("title", {}).get("runs", [{}])[0].get("text", "Unknown")
            })
    except:
        try:
            for video in scrapetube.get_channel(channel_url=f"https://www.youtube.com/@{channel_handle}", limit=limit):
                videos.append({
                    "id": video["videoId"],
                    "title": video.get("title", {}).get("runs", [{}])[0].get("text", "Unknown")
                })
        except Exception as e:
            st.error(f"Could not fetch channel: {e}")
    return videos


def get_transcript_with_timestamps(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    
    processed = []
    for item in transcript:
        processed.append({
            "text": item.text,
            "start": item.start,
            "end": item.start + item.duration
        })
    return processed


def create_documents_with_timestamps(transcript_segments, video_id, video_title, chunk_size=500):
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
                metadata={
                    "start": current_start,
                    "end": current_end,
                    "video_id": video_id,
                    "video_title": video_title,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
            )
            documents.append(doc)
            current_text = ""
            current_start = None
            current_end = None
    
    if current_text.strip():
        doc = Document(
            page_content=current_text.strip(),
            metadata={
                "start": current_start,
                "end": current_end,
                "video_id": video_id,
                "video_title": video_title,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
        )
        documents.append(doc)
    
    return documents


def semantic_search(documents, query, api_key, top_k=5):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    processed_results = []
    for doc, distance in results:
        similarity = max(0, min(100, (1 - distance / 2) * 100))
        processed_results.append((doc, similarity))
    
    return processed_results


# ============================================
# MAIN UI
# ============================================

who = st.text_input("Who to ask?", placeholder="YouTube channel URL (e.g. https://www.youtube.com/@IAmMarkManson)")
what = st.text_input("What to ask?", placeholder="Your question")
video_limit = st.slider("How many videos to search?", min_value=10, max_value=100, value=30)

if st.button("Search Channel"):
    if not who or not what:
        st.warning("Please fill in both fields.")
        st.stop()
    
    channel_handle = extract_channel_id(who)
    if not channel_handle:
        st.error("Invalid channel URL.")
        st.stop()
    
    status = st.status("Processing channel...", expanded=True)
    
    try:
        status.write(f"Fetching videos from @{channel_handle}...")
        videos = get_channel_videos(channel_handle, limit=video_limit)
        
        if not videos:
            status.update(label="Error", state="error")
            st.error("Could not find any videos. Check the channel URL.")
            st.stop()
        
        status.write(f"Found {len(videos)} videos")
        
        status.write("Fetching transcripts (this may take a minute)...")
        all_documents = []
        success_count = 0
        
        progress_bar = st.progress(0)
        
        for i, video in enumerate(videos):
            transcript = get_transcript_with_timestamps(video["id"])
            if transcript:
                docs = create_documents_with_timestamps(
                    transcript, 
                    video["id"], 
                    video["title"]
                )
                all_documents.extend(docs)
                success_count += 1
            
            progress_bar.progress((i + 1) / len(videos))
        
        progress_bar.empty()
        status.write(f"Processed {success_count} videos with captions")
        status.write(f"Created {len(all_documents)} searchable chunks")
        
        if not all_documents:
            status.update(label="Error", state="error")
            st.error("No transcripts found. The channel may not have captions enabled.")
            st.stop()
        
        status.write("Searching for relevant segments...")
        results = semantic_search(all_documents, what, api_key, top_k=5)
        avg_score = sum(score for _, score in results) / len(results)
        status.write(f"Relevance: {avg_score:.1f}%")
        
        status.update(label="Done!", state="complete", expanded=False)
        
        st.markdown("---")
        st.markdown("### Their Answer")
        st.markdown(f"**Relevance Score:** {avg_score:.1f}%")
        st.markdown(f"*Searched {success_count} videos, {len(all_documents)} chunks*")
        
        for i, (doc, score) in enumerate(results, 1):
            start_sec = int(doc.metadata['start'])
            video_url = doc.metadata['url']
            video_title = doc.metadata['video_title']
            
            st.markdown(f"---")
            st.markdown(f"**Clip {i}** from: *{video_title}*")
            st.markdown(f"Relevance: {score:.1f}% | Starts at {start_sec}s")
            st.video(video_url, start_time=start_sec)
            
            with st.expander(f"View transcript"):
                st.write(doc.page_content)
    
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>RAG + Semantic Search Across Entire Channels</p>", unsafe_allow_html=True)