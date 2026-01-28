import streamlit as st
import time
import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
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


def get_channel_videos_with_titles(channel_handle, limit=100):
    """Get video IDs and titles from a channel"""
    videos = scrapetube.get_channel(channel_username=channel_handle, limit=limit)
    video_list = []
    for video in videos:
        video_id = video.get('videoId')
        
        # Extract title from scrapetube's nested structure
        title_data = video.get('title', {})
        if isinstance(title_data, dict):
            runs = title_data.get('runs', [])
            if runs:
                title = runs[0].get('text', 'Untitled')
            else:
                title = title_data.get('simpleText', 'Untitled')
        else:
            title = str(title_data) if title_data else 'Untitled'
        
        video_list.append({
            'video_id': video_id,
            'title': title
        })
    return video_list


def select_relevant_videos(videos, question, openai_api_key, max_videos=20):
    """Use GPT 4o mini to select most relevant videos based on the question"""
    client = OpenAI(api_key=openai_api_key)
    
    # Create a numbered list of videos for the prompt
    video_list_text = "\n".join([f"{i+1}. {v['title']}" for i, v in enumerate(videos)])
    
    prompt = f"""You are helping select YouTube videos that are most likely to answer a user's question.

User's Question: {question}

Available Videos (from a YouTube channel):
{video_list_text}

Select up to {max_videos} videos that are most likely to contain relevant information for answering the question. Consider:
1. Direct topic matches
2. Related concepts that might address the question
3. General advice videos that could be relevant

Return ONLY a JSON array of video numbers (1 indexed) ordered by relevance. Example: [3, 7, 12, 1, 15]

If fewer than {max_videos} videos seem relevant, return only the relevant ones.
Return ONLY the JSON array, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up potential markdown formatting
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        
        selected_indices = json.loads(content)
        
        # Convert to 0 indexed and get actual videos
        selected_videos = []
        for idx in selected_indices:
            if 1 <= idx <= len(videos):
                selected_videos.append(videos[idx - 1])
        
        return selected_videos if selected_videos else videos[:max_videos]
    
    except Exception as e:
        # Fallback to first max_videos if AI selection fails
        st.warning(f"Smart selection failed, using first {max_videos} videos. Error: {str(e)}")
        return videos[:max_videos]


def fetch_transcript_with_timestamps(video_id, max_retries=3):
    """
    Fetch transcript using the robust list then fetch approach.
    Prefers manual captions but falls back to auto generated.
    """
    ytt = YouTubeTranscriptApi()
    
    for attempt in range(max_retries):
        try:
            # First list available transcripts
            tlist = ytt.list(video_id)
            
            # Prefer manual English captions, then any manual, then generated
            chosen = None
            
            # Priority 1: Manual English
            for t in tlist:
                code = (t.language_code or "").lower()
                if not t.is_generated and (code == "en" or code.startswith("en-")):
                    chosen = t
                    break
            
            # Priority 2: Any manual caption
            if chosen is None:
                for t in tlist:
                    if not t.is_generated:
                        chosen = t
                        break
            
            # Priority 3: Generated English
            if chosen is None:
                for t in tlist:
                    code = (t.language_code or "").lower()
                    if code == "en" or code.startswith("en-"):
                        chosen = t
                        break
            
            # Priority 4: Any available
            if chosen is None:
                chosen = next(iter(tlist), None)
            
            if chosen is None:
                return {"ok": False, "error": "No transcripts found", "segments": []}
            
            # Now fetch the actual transcript
            transcript = chosen.fetch()
            raw = transcript.to_raw_data()
            
            processed = []
            for item in raw:
                text = item.get("text", "")
                start = item.get("start", 0)
                duration = item.get("duration", 0)
                processed.append({
                    "text": text,
                    "start": start,
                    "end": start + duration,
                    "video_id": video_id
                })
            
            return {
                "ok": True,
                "segments": processed,
                "language": chosen.language_code,
                "is_generated": chosen.is_generated
            }
            
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            # These are permanent failures, no point retrying
            return {
                "ok": False,
                "error": type(e).__name__,
                "segments": []
            }
        except Exception as e:
            # Transient error, retry with backoff
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return {
                "ok": False,
                "error": str(e),
                "segments": []
            }
    
    return {"ok": False, "error": "Max retries exceeded", "segments": []}


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
        
        # If switching videos, save current chunk
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
    
    # Don't forget the last chunk
    if current_text.strip():
        doc = Document(
            page_content=current_text.strip(),
            metadata={"start": current_start, "end": current_end, "video_id": current_video}
        )
        documents.append(doc)
    
    return documents


def semantic_search(documents, query, openai_api_key, top_k=3):
    """Search for relevant chunks using embeddings"""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    processed_results = []
    for doc, distance in results:
        # Convert distance to similarity percentage
        similarity = max(0, min(100, (1 - distance / 2) * 100))
        processed_results.append((doc, similarity))
    
    return processed_results


# ============================================
# MAIN UI
# ============================================

who = st.text_input("Who to ask?", placeholder="YouTube channel URL (e.g. youtube.com/@MarkManson)")
what = st.text_input("What to ask?", placeholder="Your question")

if st.button("Generate Answer"):
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
            # STEP 1: Fetch all video titles from channel
            status.write(f"📺 Fetching video list from @{channel_handle}...")
            all_videos = get_channel_videos_with_titles(channel_handle, limit=100)
            status.write(f"   Found {len(all_videos)} videos in channel")
            
            if not all_videos:
                st.error("Could not find any videos in this channel.")
                st.stop()
            
            # STEP 2: Use AI to select most relevant videos
            status.write("🧠 Analyzing titles to find relevant videos...")
            selected_videos = select_relevant_videos(all_videos, what, api_key, max_videos=20)
            status.write(f"   Selected {len(selected_videos)} relevant videos")
            
            # Show which videos were selected
            with st.expander("Selected Videos", expanded=False):
                for i, v in enumerate(selected_videos, 1):
                    st.write(f"{i}. {v['title']}")
            
            # STEP 3: Fetch transcripts for selected videos
            status.write("📝 Fetching transcripts...")
            successful = 0
            failed = 0
            
            for i, video in enumerate(selected_videos, 1):
                vid = video['video_id']
                title_short = video['title'][:40] + "..." if len(video['title']) > 40 else video['title']
                status.write(f"   [{i}/{len(selected_videos)}] {title_short}")
                
                result = fetch_transcript_with_timestamps(vid)
                
                if result["ok"]:
                    all_transcripts.extend(result["segments"])
                    successful += 1
                else:
                    failed += 1
                
                # Small delay between requests to be nice to YouTube
                if i < len(selected_videos):
                    time.sleep(0.5)
            
            status.write(f"   ✓ Got transcripts from {successful} videos ({failed} unavailable)")
            status.write(f"   Total segments: {len(all_transcripts)}")
        
        else:
            # Single video mode
            status.write("📝 Fetching transcript...")
            result = fetch_transcript_with_timestamps(single_video_id)
            
            if result["ok"]:
                all_transcripts = result["segments"]
                status.write(f"   Found {len(all_transcripts)} segments")
            else:
                st.error(f"Could not get transcript: {result.get('error', 'Unknown error')}")
                st.stop()
        
        if not all_transcripts:
            st.error("No transcripts found. The selected videos may not have captions available.")
            st.stop()
        
        # STEP 4: Create searchable chunks
        status.write("🔧 Creating searchable chunks...")
        documents = create_documents_with_timestamps(all_transcripts)
        status.write(f"   Created {len(documents)} chunks")
        
        # STEP 5: Semantic search
        status.write("🔍 Searching for relevant segments...")
        results = semantic_search(documents, what, api_key, top_k=3)
        avg_score = sum(score for _, score in results) / len(results) if results else 0
        status.write(f"   Relevance: {avg_score:.1f}%")
        
        status.update(label="Done!", state="complete", expanded=False)
        
        # Display results
        st.markdown("---")
        st.markdown("### Their Answer")
        st.markdown(f"**Relevance Score:** {avg_score:.1f}%")
        
        for i, (doc, score) in enumerate(results, 1):
            start_sec = int(doc.metadata['start'])
            video_id = doc.metadata['video_id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            st.markdown(f"**Clip {i}** (starts at {start_sec}s)")
            st.video(video_url, start_time=start_sec)
            with st.expander(f"Transcript for clip {i}"):
                st.write(doc.page_content)
    
    except Exception as e:
        status.update(label="Error", state="error")
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>Smart Video Selection + RAG + Semantic Search</p>", unsafe_allow_html=True)
