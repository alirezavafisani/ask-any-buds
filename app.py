import streamlit as st
import time
import json
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
st.markdown("<p style='text-align:center;color:#666;'>Agentic RAG | Ask any YouTuber anything</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================
# API KEY & LLM
# ============================================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key missing.")
    st.stop()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

# ============================================
# CORE FUNCTIONS (ORIGINAL WORKING VERSIONS)
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
            title = "Unknown"
            try:
                title = video.get("title", {}).get("runs", [{}])[0].get("text", "Unknown")
            except:
                pass
            videos.append({
                "id": video["videoId"],
                "title": title
            })
    except:
        try:
            for video in scrapetube.get_channel(channel_url=f"https://www.youtube.com/@{channel_handle}", limit=limit):
                title = "Unknown"
                try:
                    title = video.get("title", {}).get("runs", [{}])[0].get("text", "Unknown")
                except:
                    pass
                videos.append({
                    "id": video["videoId"],
                    "title": title
                })
        except:
            pass
    return videos


def get_transcript_with_timestamps(video_id):
    """ORIGINAL WORKING VERSION - DO NOT CHANGE"""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        processed = []
        for item in transcript_list:
            processed.append({
                "text": item.text,
                "start": item.start,
                "end": item.start + item.duration
            })
        return processed
    except:
        return None


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


# ============================================
# AGENT FUNCTIONS
# ============================================

def rewrite_query(original_query, feedback=None):
    """Agent: Query Rewriter"""
    feedback_text = f"\nPrevious search failed. Feedback: {feedback}" if feedback else ""
    
    prompt = f"""Generate 3 different search queries to find video clips answering this question:
"{original_query}"
{feedback_text}

Return ONLY a JSON array of 3 strings. Example: ["query one", "query two", "query three"]"""

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)[:3]
    except:
        return [original_query]


def rank_videos_by_relevance(videos, question):
    """Agent: Video Ranker"""
    video_list = "\n".join([f"{i}. {v['title']}" for i, v in enumerate(videos)])
    
    prompt = f"""Question: "{question}"

Videos:
{video_list}

Return JSON array of video numbers ordered by relevance (most relevant first).
Example: [5, 2, 8, 0, 1]"""

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        rankings = json.loads(content)
        return [int(r) for r in rankings if int(r) < len(videos)]
    except:
        return list(range(len(videos)))


def evaluate_chunks(question, chunks):
    """Agent: Evaluator"""
    chunks_text = "\n\n".join([f"Chunk {i+1}: {doc.page_content[:400]}" for i, (doc, _) in enumerate(chunks[:5])])
    
    prompt = f"""Question: "{question}"

Retrieved chunks:
{chunks_text}

Return JSON: {{"is_good": true/false, "score": 0-100, "feedback": "what to search for if not good"}}"""

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        return json.loads(content)
    except:
        return {"is_good": True, "score": 70, "feedback": ""}


def rerank_chunks(question, chunks):
    """Agent: Re-ranker"""
    chunks_text = "\n\n".join([f"[{i}] {doc.page_content[:300]}" for i, (doc, _) in enumerate(chunks)])
    
    prompt = f"""Question: "{question}"

Chunks:
{chunks_text}

Score each chunk 0-100 for relevance. Return JSON: {{"0": 85, "1": 45, "2": 72}}"""

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        scores = json.loads(content)
        reranked = [(doc, float(scores.get(str(i), score))) for i, (doc, score) in enumerate(chunks)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except:
        return chunks


def semantic_search_multi_query(documents, queries, api_key, top_k=10):
    """Search with multiple queries"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    
    all_results = {}
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        for doc, distance in results:
            doc_id = f"{doc.metadata['video_id']}_{doc.metadata['start']}"
            similarity = max(0, min(100, (1 - distance / 2) * 100))
            if doc_id not in all_results or similarity > all_results[doc_id][1]:
                all_results[doc_id] = (doc, similarity)
    
    combined = list(all_results.values())
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


# ============================================
# MAIN UI
# ============================================

who = st.text_input("Who to ask?", placeholder="YouTube channel URL")
what = st.text_input("What to ask?", placeholder="Your question")
video_limit = st.slider("Videos to fetch", 20, 150, 50)
top_n = st.slider("Top N relevant videos to search", 5, 30, 12)

if st.button("🎙️ Search Channel"):
    if not who or not what:
        st.warning("Please fill in both fields.")
        st.stop()
    
    channel_handle = extract_channel_id(who)
    status = st.status("🤖 Agentic RAG Pipeline...", expanded=True)
    
    try:
        # Step 1: Get videos
        status.write(f"📺 Fetching videos from @{channel_handle}...")
        videos = get_channel_videos(channel_handle, limit=video_limit)
        
        if not videos:
            st.error("Could not find videos.")
            st.stop()
        
        status.write(f"   Found {len(videos)} videos")
        
        # Step 2: Rank videos
        status.write("🧠 Agent 1: Ranking videos by relevance...")
        rankings = rank_videos_by_relevance(videos, what)
        top_videos = [videos[i] for i in rankings[:top_n] if i < len(videos)]
        
        if not top_videos:
            top_videos = videos[:top_n]
        
        status.write(f"   Selected top {len(top_videos)} videos")
        
        # Step 3: Fetch transcripts
        status.write("📝 Fetching transcripts...")
        all_documents = []
        success_count = 0
        
        progress = st.progress(0)
        for i, video in enumerate(top_videos):
            transcript = get_transcript_with_timestamps(video["id"])
            if transcript:
                docs = create_documents_with_timestamps(transcript, video["id"], video["title"])
                all_documents.extend(docs)
                success_count += 1
                status.write(f"   ✓ {video['title'][:50]}")
            progress.progress((i + 1) / len(top_videos))
            time.sleep(0.3)
        
        progress.empty()
        
        # Fallback if no transcripts
        if not all_documents:
            status.write("⚠️ Trying remaining videos...")
            remaining = [v for v in videos if v not in top_videos]
            for video in remaining[:20]:
                transcript = get_transcript_with_timestamps(video["id"])
                if transcript:
                    docs = create_documents_with_timestamps(transcript, video["id"], video["title"])
                    all_documents.extend(docs)
                    success_count += 1
                    status.write(f"   ✓ {video['title'][:50]}")
                    if success_count >= 8:
                        break
                time.sleep(0.3)
        
        status.write(f"   Total: {success_count} videos, {len(all_documents)} chunks")
        
        if not all_documents:
            st.error("No transcripts found. Try a different channel.")
            st.stop()
        
        # Step 4: Agentic search loop
        best_results = None
        best_score = 0
        feedback = None
        
        for attempt in range(3):
            status.write(f"🔄 Search attempt {attempt + 1}/3")
            
            # Rewrite query
            status.write("   🧠 Agent 2: Rewriting query...")
            queries = rewrite_query(what, feedback)
            status.write(f"   Queries: {queries}")
            
            # Search
            results = semantic_search_multi_query(all_documents, queries, api_key, top_k=8)
            
            # Evaluate
            status.write("   🧠 Agent 3: Evaluating results...")
            evaluation = evaluate_chunks(what, results)
            status.write(f"   Score: {evaluation['score']}%, Good: {evaluation['is_good']}")
            
            if evaluation["score"] > best_score:
                best_score = evaluation["score"]
                best_results = results
            
            if evaluation["is_good"] or attempt == 2:
                break
            
            feedback = evaluation["feedback"]
        
        # Step 5: Re-rank
        status.write("🧠 Agent 4: Re-ranking results...")
        final_results = rerank_chunks(what, best_results[:6])
        final_score = sum(s for _, s in final_results) / len(final_results)
        
        status.update(label="✅ Complete!", state="complete", expanded=False)
        
        # Display
        st.markdown("---")
        st.markdown("### 🎧 Their Answer")
        st.markdown(f"**Relevance Score:** {final_score:.1f}%")
        st.markdown(f"*{success_count} videos, {len(all_documents)} chunks, {attempt + 1} iterations*")
        
        for i, (doc, score) in enumerate(final_results[:5], 1):
            start_sec = int(doc.metadata['start'])
            st.markdown("---")
            st.markdown(f"**Clip {i}** from: *{doc.metadata['video_title']}*")
            st.markdown(f"Relevance: {score:.1f}% | Starts at {start_sec}s")
            st.video(doc.metadata['url'], start_time=start_sec)
            with st.expander("View transcript"):
                st.write(doc.page_content)
    
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>Agentic RAG Pipeline</p>", unsafe_allow_html=True)