import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
import scrapetube
import json

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
    .agent-step {
        background: #1a1a2e;
        border-left: 3px solid #4a9eff;
        padding: 10px;
        margin: 5px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# AskAnyBuds")
st.markdown("<p style='text-align:center;color:#666;'>Agentic RAG | Ask any YouTuber anything</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================
# API KEY & LLM SETUP
# ============================================
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key missing. Add it to Streamlit Secrets.")
    st.stop()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
llm_judge = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# ============================================
# AGENT FUNCTIONS
# ============================================

def rewrite_query(original_query, feedback=None):
    """
    Agent 1: Query Rewriter
    Expands user question into multiple semantic search queries
    """
    feedback_text = ""
    if feedback:
        feedback_text = f"\n\nPrevious queries didn't find good results. Feedback: {feedback}\nTry different angles and synonyms."
    
    prompt = f"""You are a search query optimizer. The user wants to find video clips where a YouTuber answers their question.

Original question: "{original_query}"
{feedback_text}

Generate 3 different search queries that would help find relevant video segments. Each query should:
1. Use different phrasings and synonyms
2. Focus on key concepts the YouTuber might discuss
3. Be 5 to 15 words long

Return ONLY a JSON array of 3 strings, nothing else.
Example: ["query one here", "query two here", "query three here"]"""

    response = llm.invoke(prompt)
    try:
        queries = json.loads(response.content)
        return queries[:3]
    except:
        return [original_query]


def rank_videos_by_relevance(videos, question):
    """
    Agent 2: Video Ranker
    Ranks videos by title/description relevance to the question
    """
    video_list = ""
    for i, v in enumerate(videos):
        title = v.get("title", "Unknown")
        video_list += f"{i}. {title}\n"
    
    prompt = f"""Given this question: "{question}"

And these video titles:
{video_list}

Return a JSON array of the video numbers (integers) ordered from MOST relevant to LEAST relevant.
Only return the JSON array, nothing else. Example: [5, 2, 8, 0, 1, 3, 4, 6, 7]"""

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        rankings = json.loads(content)
        rankings = [int(r) for r in rankings if int(r) < len(videos)]
        if not rankings:
            rankings = list(range(len(videos)))
        return rankings
    except:
        return list(range(len(videos)))


def evaluate_chunks(question, chunks, attempt_number):
    """
    Agent 3: Chunk Evaluator
    Judges if retrieved chunks actually answer the question
    Returns (is_good_enough, true_relevance_score, feedback_for_rewrite)
    """
    chunks_text = ""
    for i, (doc, score) in enumerate(chunks):
        chunks_text += f"Chunk {i+1}:\n{doc.page_content[:500]}\n\n"
    
    prompt = f"""You are a strict relevance judge. Evaluate if these transcript chunks actually answer the user's question.

User's Question: "{question}"

Retrieved Chunks:
{chunks_text}

Evaluate:
1. Do these chunks DIRECTLY address the question? (not tangentially related)
2. Would a user be satisfied with these as "the answer"?
3. What is the TRUE relevance score (0 to 100)?

Return ONLY a JSON object:
{{
    "is_good_enough": true/false,
    "true_relevance": 0-100,
    "feedback": "If not good enough, explain what kind of content we should search for instead"
}}"""

    response = llm_judge.invoke(prompt)
    try:
        result = json.loads(response.content)
        return result
    except:
        return {"is_good_enough": True, "true_relevance": 70, "feedback": ""}


def rerank_chunks(question, chunks):
    """
    Agent 4: Re-ranker
    Uses LLM to re-score and reorder chunks by true relevance
    """
    chunks_text = ""
    for i, (doc, score) in enumerate(chunks):
        chunks_text += f"[{i}] {doc.page_content[:400]}\n\n"
    
    prompt = f"""You are a relevance re-ranker. Given a question and retrieved text chunks, score each chunk's relevance.

Question: "{question}"

Chunks:
{chunks_text}

For each chunk, assign a relevance score from 0 to 100 based on how directly it answers the question.

Return ONLY a JSON object mapping chunk index to score:
{{"0": 85, "1": 45, "2": 72, ...}}"""

    response = llm_judge.invoke(prompt)
    try:
        scores = json.loads(response.content)
        reranked = []
        for i, (doc, old_score) in enumerate(chunks):
            new_score = scores.get(str(i), old_score)
            reranked.append((doc, float(new_score)))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except:
        return chunks


# ============================================
# CORE FUNCTIONS
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
            
            description = ""
            try:
                description = video.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", "")
            except:
                pass
            
            videos.append({
                "id": video["videoId"],
                "title": title,
                "description": description
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
                    "title": title,
                    "description": ""
                })
        except Exception as e:
            st.error(f"Could not fetch channel: {e}")
    return videos


def get_transcript_with_timestamps(video_id):
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
    except Exception as e:
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


def semantic_search_multi_query(documents, queries, api_key, top_k=10):
    """Search with multiple queries and combine results"""
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

who = st.text_input("Who to ask?", placeholder="YouTube channel URL (e.g. https://www.youtube.com/@IAmMarkManson)")
what = st.text_input("What to ask?", placeholder="Your question")
video_limit = st.slider("Videos to fetch from channel", min_value=20, max_value=150, value=50)
top_n_videos = st.slider("Top N relevant videos to search", min_value=5, max_value=30, value=12)

if st.button("🎙️ Search Channel"):
    if not who or not what:
        st.warning("Please fill in both fields.")
        st.stop()
    
    channel_handle = extract_channel_id(who)
    if not channel_handle:
        st.error("Invalid channel URL.")
        st.stop()
    
    status = st.status("🤖 Agentic RAG Pipeline Running...", expanded=True)
    
    try:
        # Step 1: Fetch videos
        status.write(f"📺 Fetching videos from @{channel_handle}...")
        videos = get_channel_videos(channel_handle, limit=video_limit)
        
        if not videos:
            status.update(label="❌ Error", state="error")
            st.error("Could not find any videos.")
            st.stop()
        
        status.write(f"   Found {len(videos)} videos")
        
        # Step 2: Agent ranks videos
        status.write("🧠 Agent 1: Ranking videos by relevance to your question...")
        rankings = rank_videos_by_relevance(videos, what)
        
        top_videos = []
        for i in rankings[:top_n_videos]:
            try:
                if isinstance(i, int) and 0 <= i < len(videos):
                    top_videos.append(videos[i])
            except:
                continue

        if not top_videos:
            top_videos = videos[:top_n_videos]
        status.write(f"   Selected top {len(top_videos)} most relevant videos")
        
        # Step 3: Fetch transcripts for top videos only
        status.write("📝 Fetching transcripts from relevant videos...")
        all_documents = []
        success_count = 0
        failed_videos = []
        
        progress_bar = st.progress(0)
        
        for i, video in enumerate(top_videos):
            transcript = get_transcript_with_timestamps(video["id"])
            if transcript:
                docs = create_documents_with_timestamps(
                    transcript, 
                    video["id"], 
                    video["title"]
                )
                all_documents.extend(docs)
                success_count += 1
                status.write(f"   ✓ {video['title'][:50]}")
            else:
                failed_videos.append(video["id"])
            progress_bar.progress((i + 1) / len(top_videos))
        
        progress_bar.empty()
        
        # Fallback: if no transcripts from ranked videos, try remaining videos
        if success_count == 0:
            status.write("⚠️ Ranked videos had no captions. Trying other videos...")
            remaining = [v for v in videos if v["id"] not in [tv["id"] for tv in top_videos]]
            
            for i, video in enumerate(remaining[:20]):
                transcript = get_transcript_with_timestamps(video["id"])
                if transcript:
                    docs = create_documents_with_timestamps(
                        transcript, 
                        video["id"], 
                        video["title"]
                    )
                    all_documents.extend(docs)
                    success_count += 1
                    status.write(f"   ✓ {video['title'][:50]}")
                    if success_count >= 10:
                        break
        
        status.write(f"   Processed {success_count} videos, {len(all_documents)} chunks")
        
        if not all_documents:
            status.update(label="❌ Error", state="error")
            st.error("No transcripts found.")
            st.stop()
        
        # Step 4: Agentic search loop (up to 3 attempts)
        best_results = None
        best_score = 0
        final_feedback = None
        
        for attempt in range(3):
            status.write(f"🔄 Attempt {attempt + 1}/3")
            
            # Agent rewrites query
            feedback = final_feedback if attempt > 0 else None
            status.write(f"   🧠 Agent 2: Rewriting query...")
            queries = rewrite_query(what, feedback)
            status.write(f"   Generated queries: {queries}")
            
            # Search
            status.write(f"   🔍 Searching with {len(queries)} queries...")
            results = semantic_search_multi_query(all_documents, queries, api_key, top_k=8)
            
            # Agent evaluates
            status.write(f"   🧠 Agent 3: Evaluating results...")
            evaluation = evaluate_chunks(what, results, attempt)
            
            status.write(f"   Evaluation: relevance={evaluation['true_relevance']}%, good_enough={evaluation['is_good_enough']}")
            
            if evaluation["true_relevance"] > best_score:
                best_score = evaluation["true_relevance"]
                best_results = results
            
            if evaluation["is_good_enough"] or attempt == 2:
                break
            
            final_feedback = evaluation["feedback"]
            status.write(f"   Feedback: {final_feedback}")
        
        # Step 5: Re-rank final results
        status.write("🧠 Agent 4: Re-ranking final results...")
        final_results = rerank_chunks(what, best_results[:6])
        
        final_avg = sum(score for _, score in final_results) / len(final_results)
        status.write(f"   Final relevance: {final_avg:.1f}%")
        
        status.update(label="✅ Agentic Pipeline Complete!", state="complete", expanded=False)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎧 Their Answer")
        st.markdown(f"**True Relevance Score:** {final_avg:.1f}%")
        st.markdown(f"*Searched {success_count} videos, {len(all_documents)} chunks, {attempt + 1} search iterations*")
        
        for i, (doc, score) in enumerate(final_results[:5], 1):
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
        status.update(label="❌ Error", state="error")
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#444;font-size:12px;'>Agentic RAG | Query Rewriting | Video Ranking | Evaluation Loops | Re-ranking</p>", unsafe_allow_html=True)