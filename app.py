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


def extract_channel_handle(url):
   """Extract channel handle from URL like youtube.com/@MarkManson"""
   if "/@" in url:
       handle = url.split("/@")[1].split("/")[0].split("?")[0]
       return handle
   elif "/channel/" in url:
       return url.split("/channel/")[1].split("/")[0].split("?")[0]
   elif "/c/" in url:
       return url.split("/c/")[1].split("/")[0].split("?")[0]
   return None




def extract_video_id(url):
   """Extract video ID from a single video URL"""
   if "v=" in url:
       return url.split("v=")[1].split("&")[0]
   elif "youtu.be/" in url:
       return url.split("youtu.be/")[1].split("?")[0]
   elif "shorts/" in url:
       return url.split("shorts/")[1].split("?")[0]
   return None




def get_channel_videos(channel_handle, limit=10):
   """Get list of video IDs from a channel"""
   videos = scrapetube.get_channel(channel_username=channel_handle, limit=limit)
   video_ids = []
   for video in videos:
       video_ids.append(video['videoId'])
   return video_ids




def get_transcript_with_timestamps(video_id):
   """Fetch transcript for a single video"""
   try:
       ytt_api = YouTubeTranscriptApi()
       transcript_list = ytt_api.fetch(video_id)
       processed = []
       for item in transcript_list:
           processed.append({
               "text": item.text,
               "start": item.start,
               "end": item.start + item.duration,
               "video_id": video_id
           })
       return processed
   except Exception as e:
       return []




def create_documents_with_timestamps(transcript_segments, chunk_size=500):
   """Create documents preserving timestamps and video source"""
   documents = []
   current_text = ""
   current_start = None
   current_end = None
   current_video = None
  
   for seg in transcript_segments:
       if current_start is None:
           current_start = seg["start"]
           current_video = seg["video_id"]
      
       if seg["video_id"] != current_video:
           if current_text.strip():
               doc = Document(
                   page_content=current_text.strip(),
                   metadata={"start": current_start, "end": current_end, "video_id": current_video}
               )
               documents.append(doc)
           current_text = ""
           current_start = seg["start"]
           current_video = seg["video_id"]
      
       current_text += " " + seg["text"]
       current_end = seg["end"]
      
       if len(current_text) >= chunk_size:
           doc = Document(
               page_content=current_text.strip(),
               metadata={"start": current_start, "end": current_end, "video_id": current_video}
           )
           documents.append(doc)
           current_text = ""
           current_start = None
           current_end = None
  
   if current_text.strip():
       doc = Document(
           page_content=current_text.strip(),
           metadata={"start": current_start, "end": current_end, "video_id": current_video}
       )
       documents.append(doc)
  
   return documents




def semantic_search(documents, query, api_key, top_k=3):
   """Search for relevant chunks"""
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


who = st.text_input("Who to ask?", placeholder="YouTube channel URL (e.g. youtube.com/@MarkManson)")
num_videos = st.slider("How many videos to search?", min_value=1, max_value=50, value=5)
what = st.text_input("What to ask?", placeholder="Your question")


if st.button("🎙️ Generate Answer"):
   if not who or not what:
       st.warning("Please fill in both fields.")
       st.stop()
  
   # Check if it's a channel or single video
   channel_handle = extract_channel_handle(who)
   single_video_id = extract_video_id(who)
  
   if not channel_handle and not single_video_id:
       st.error("Invalid URL. Provide a channel or video URL.")
       st.stop()
  
   status = st.status("Processing...", expanded=True)
  
   try:
       all_transcripts = []
      
       if channel_handle:
           status.write(f"📺 Fetching videos from @{channel_handle}...")
           video_ids = get_channel_videos(channel_handle, limit=num_videos)
           status.write(f"   Found {len(video_ids)} videos")
          
           for i, vid in enumerate(video_ids, 1):
               status.write(f"📝 Getting transcript {i}/{len(video_ids)}...")
               transcript = get_transcript_with_timestamps(vid)
               if transcript:
                   all_transcripts.extend(transcript)
          
           status.write(f"   Total segments: {len(all_transcripts)}")
       else:
           status.write("📝 Fetching transcript...")
           all_transcripts = get_transcript_with_timestamps(single_video_id)
           for seg in all_transcripts:
               seg["video_id"] = single_video_id
           status.write(f"   Found {len(all_transcripts)} segments")
      
       if not all_transcripts:
           st.error("No transcripts found. Videos may not have captions.")
           st.stop()
      
       status.write("📦 Creating chunks...")
       documents = create_documents_with_timestamps(all_transcripts)
       status.write(f"   Created {len(documents)} chunks")
      
       status.write("🔍 Searching for relevant segments...")
       results = semantic_search(documents, what, api_key, top_k=3)
       avg_score = sum(score for _, score in results) / len(results)
       status.write(f"   Relevance: {avg_score:.1f}%")
      
       status.update(label="✅ Done!", state="complete", expanded=False)
      
       st.markdown("---")
       st.markdown("### 🎧 Their Answer")
       st.markdown(f"**Relevance Score:** {avg_score:.1f}%")
      
       for i, (doc, score) in enumerate(results, 1):
           start_sec = int(doc.metadata['start'])
           video_id = doc.metadata['video_id']
           video_url = f"https://www.youtube.com/watch?v={video_id}"
          
           st.markdown(f"**Clip {i}** (from video, starts at {start_sec}s)")
           st.video(video_url, start_time=start_sec)
           with st.expander(f"Transcript for clip {i}"):
               st.write(doc.page_content)
  
   except Exception as e:
       status.update(label="❌ Error", state="error")
       st.error(f"Error: {str(e)}")
       st.info("Make sure the channel exists and videos have captions.")


st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>Multi-Video RAG + Semantic Search</p>", unsafe_allow_html=True)
