import streamlit as st
import time
import json
import requests
import random
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from openai import OpenAI

# YouTube Data API endpoints
YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

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
# API KEYS
# ============================================
openai_api_key = st.secrets.get("OPENAI_API_KEY")
youtube_api_key = st.secrets.get("YOUTUBE_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key missing. Add it to Streamlit Secrets.")
    st.stop()

if not youtube_api_key:
    st.error("YouTube API Key missing. Add YOUTUBE_API_KEY to Streamlit Secrets.")
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


def get_channel_id_from_handle(handle, youtube_api_key):
    """Convert @handle to channel ID using YouTube Data API"""
    # Try as custom URL first
    params = {
        "key": youtube_api_key,
        "part": "id",
        "forHandle": handle
    }
    r = requests.get(YT_CHANNELS_URL, params=params, timeout=20)
    
    if r.status_code == 200:
        data = r.json()
        items = data.get("items", [])
        if items:
            return items[0]["id"]
    
    # Fallback: search for the channel
    params = {
        "key": youtube_api_key,
        "part": "snippet",
        "q": handle,
        "type": "channel",
        "maxResults": 1
    }
    r = requests.get(YT_SEARCH_URL, params=params, timeout=20)
    if r.status_code == 200:
        data = r.json()
        items = data.get("items", [])
        if items:
            return items[0]["snippet"]["channelId"]
    
    return None


def get_channel_videos_with_titles(channel_handle, youtube_api_key, limit=50):
    """Get video IDs and titles from a channel using YouTube Data API"""
    
    # First get the channel ID
    channel_id = get_channel_id_from_handle(channel_handle, youtube_api_key)
    if not channel_id:
        return []
    
    video_list = []
    next_page_token = None
    
    while len(video_list) < limit:
        # Search for videos from this channel
        params = {
            "key": youtube_api_key,
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",  # Most recent first
            "maxResults": min(50, limit - len(video_list)),
        }
        if next_page_token:
            params["pageToken"] = next_page_token
        
        r = requests.get(YT_SEARCH_URL, params=params, timeout=20)
        if r.status_code != 200:
            break
        
        data = r.json()
        
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            title = item.get("snippet", {}).get("title", "Untitled")
            if video_id:
                video_list.append({
                    "video_id": video_id,
                    "title": title
                })
        
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
    
    return video_list[:limit]


def select_relevant_videos(videos, question, openai_key, max_videos=20):
    """Use GPT 4o mini to select most relevant videos based on the question"""
    client = OpenAI(api_key=openai_key)
    
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


def fetch_transcript_with_timestamps(video_id, max_retries=2):
    """
    Fetch transcript using the robust list then fetch approach.
    Prefers manual captions but falls back to auto generated.
    """
    ytt = YouTubeTranscriptApi()
    
    for attempt in range(max_retries):
        try:
            # First list available transcripts
            tlist = ytt.list(video_id)
            
            # Convert to list to check if empty
            transcript_options = list(tlist)
            if not transcript_options:
                return {"ok": False, "error": "No transcript tracks available", "segments": []}
            
            # Prefer manual English captions, then any manual, then generated
            chosen = None
            
            # Priority 1: Manual English
            for t in transcript_options:
                code = (t.language_code or "").lower()
                if not t.is_generated and (code == "en" or code.startswith("en-")):
                    chosen = t
                    break
            
            # Priority 2: Any manual caption
            if chosen is None:
                for t in transcript_options:
                    if not t.is_generated:
                        chosen = t
                        break
            
            # Priority 3: Generated English
            if chosen is None:
                for t in transcript_options:
                    code = (t.language_code or "").lower()
                    if code == "en" or code.startswith("en-"):
                        chosen = t
                        break
            
            # Priority 4: Any available
            if chosen is None:
                chosen = transcript_options[0]
            
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
            
        except TranscriptsDisabled:
            return {"ok": False, "error": "TranscriptsDisabled", "segments": []}
        except NoTranscriptFound:
            return {"ok": False, "error": "NoTranscriptFound", "segments": []}
        except VideoUnavailable:
            return {"ok": False, "error": "VideoUnavailable", "segments": []}
        except Exception as e:
            error_str = str(e)
            # Check if it's an IP block
            if "RequestBlocked" in error_str or "blocked" in error_str.lower():
                # For IP blocks, retry with longer delay
                if attempt < max_retries - 1:
                    time.sleep(10 + random.uniform(0, 5))  # 10 to 15 seconds
                    continue
                return {"ok": False, "error": "IP_BLOCKED", "segments": []}
            
            # Other transient errors
            if attempt < max_retries - 1:
                time.sleep(3 + random.uniform(0, 2))
                continue
            return {
                "ok": False,
                "error": f"Exception: {type(e).__name__}: {str(e)[:100]}",
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
            # STEP 1: Fetch all video titles from channel using YouTube Data API
            status.write(f"📺 Fetching video list from @{channel_handle}...")
            all_videos = get_channel_videos_with_titles(channel_handle, youtube_api_key, limit=50)
            status.write(f"   Found {len(all_videos)} videos in channel")
            
            if not all_videos:
                st.error("Could not find any videos in this channel. Check the URL or try again.")
                st.stop()
            
            # STEP 2: Use AI to select most relevant videos
            status.write("🧠 Analyzing titles to find relevant videos...")
            selected_videos = select_relevant_videos(all_videos, what, openai_api_key, max_videos=15)
            status.write(f"   Selected {len(selected_videos)} relevant videos")
            
            # Show which videos were selected
            with st.expander("Selected Videos", expanded=False):
                for i, v in enumerate(selected_videos, 1):
                    st.write(f"{i}. {v['title']}")
            
            # STEP 3: Fetch transcripts for selected videos with long delays
            status.write("📝 Fetching transcripts (this takes a while to avoid rate limits)...")
            successful = 0
            failed = 0
            failed_details = []
            ip_blocked = False
            
            for i, video in enumerate(selected_videos, 1):
                vid = video['video_id']
                title_short = video['title'][:40] + "..." if len(video['title']) > 40 else video['title']
                status.write(f"   [{i}/{len(selected_videos)}] {title_short}")
                
                result = fetch_transcript_with_timestamps(vid)
                
                if result["ok"]:
                    all_transcripts.extend(result["segments"])
                    successful += 1
                    status.write(f"      ✓ Got {len(result['segments'])} segments")
                else:
                    failed += 1
                    error_msg = result.get("error", "Unknown")
                    failed_details.append(f"{title_short}: {error_msg}")
                    status.write(f"      ✗ Failed: {error_msg}")
                    
                    # If IP is blocked, stop trying more videos
                    if error_msg == "IP_BLOCKED":
                        ip_blocked = True
                        status.write("   ⚠️ YouTube is blocking requests. Stopping to avoid further blocks.")
                        break
                
                # Long delay between requests (5 to 8 seconds)
                if i < len(selected_videos) and not ip_blocked:
                    delay = 5 + random.uniform(0, 3)
                    status.write(f"      ⏳ Waiting {delay:.1f}s...")
                    time.sleep(delay)
            
            status.write(f"   ✓ Got transcripts from {successful} videos ({failed} unavailable)")
            
            # Show failed videos details
            if failed_details:
                with st.expander(f"Failed Videos ({len(failed_details)})", expanded=False):
                    for detail in failed_details:
                        st.write(detail)
            
            # If IP was blocked and we got nothing, show helpful message
            if ip_blocked and successful == 0:
                st.error("YouTube is blocking transcript requests from this server's IP address.")
                st.info("**Workarounds:**\n\n1. Try again later (YouTube may unblock after some time)\n\n2. Run the app locally on your computer\n\n3. Use a proxy service")
                st.stop()
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
        results = semantic_search(documents, what, openai_api_key, top_k=3)
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

# Diagnostic section
with st.expander("🔧 Diagnostic Test"):
    st.write("Test if transcript fetching works on your Streamlit instance")
    test_video_id = st.text_input("Test Video ID", value="dQw4w9WgXcQ", help="Enter any YouTube video ID to test")
    if st.button("Run Diagnostic"):
        st.write(f"Testing video ID: {test_video_id}")
        st.write(f"URL: https://youtube.com/watch?v={test_video_id}")
        
        result = fetch_transcript_with_timestamps(test_video_id)
        
        if result["ok"]:
            st.success(f"✓ SUCCESS! Got {len(result['segments'])} segments")
            st.write(f"Language: {result.get('language', 'unknown')}")
            st.write(f"Auto generated: {result.get('is_generated', 'unknown')}")
            st.write("First 3 segments:")
            for seg in result["segments"][:3]:
                st.write(f"  [{seg['start']:.1f}s] {seg['text']}")
        else:
            st.error(f"✗ FAILED: {result.get('error', 'Unknown error')}")
