import streamlit as st
import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tavily import TavilyClient

# ---------------------------------------------------------
# 1. Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hybrid AI Fake News Detector (BERT + RAG)",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Sidebar Configuration (API Key & Model Info)
# ---------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

default_api_key = os.environ.get("TAVILY_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Tavily API Key",
    value=default_api_key,
    type="password",
    help="Enter your free Tavily Search API key from https://tavily.com/"
)

# Active key preference: sidebar input -> env var -> placeholder
TAVILY_API_KEY = api_key_input.strip() if api_key_input.strip() else default_api_key

st.sidebar.divider()
st.sidebar.title("📌 Project Details")
st.sidebar.info("""
**Developer:** Gurudatta Pradhan  
**College:** Presidency University  
**Department:** AI & ML  
**Architecture:** DistilBERT + RAG (Tavily AI)  
**Verification:** Linguistic Style + Real-Time Fact-Checking
""")

# ---------------------------------------------------------
# 3. Model Loading with Fallback & CPU/CUDA Auto-Detection
# ---------------------------------------------------------
@st.cache_resource
def load_bert_model():
    """
    Loads fine-tuned DistilBERT model.
    Handles CUDA/CPU detection and falls back safely if weights are missing locally.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_name = "fine_tuned_bert_fake_news"
    model_path = os.path.join(base_dir, model_name)
    
    # Check if local weight files actually exist
    has_local_weights = os.path.exists(model_path) and any(
        os.path.exists(os.path.join(model_path, fname))
        for fname in ["model.safetensors", "pytorch_model.bin"]
    )
    
    try:
        if has_local_weights:
            tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            mod = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
            status_msg = f"Loaded fine-tuned model from `{model_name}` on {device.upper()}"
        else:
            # Fallback to standard base model if weights were not committed to git
            fallback_name = "distilbert-base-uncased"
            tok = AutoTokenizer.from_pretrained(fallback_name)
            mod = AutoModelForSequenceClassification.from_pretrained(fallback_name, num_labels=2).to(device)
            status_msg = f"Local weights not found in `{model_name}`. Using `{fallback_name}` on {device.upper()}."
    except Exception as e:
        fallback_name = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(fallback_name)
        mod = AutoModelForSequenceClassification.from_pretrained(fallback_name, num_labels=2).to(device)
        status_msg = f"Notice: {str(e)}. Initialized `{fallback_name}` on {device.upper()}."

    return tok, mod, device, status_msg

tokenizer, model, device, model_status = load_bert_model()

# ---------------------------------------------------------
# 4. Helper Functions: Query Extraction & Evidence Analysis
# ---------------------------------------------------------
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot',
    'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d',
    'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s',
    'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
    'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d',
    'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves', 'breaking', 'news', 'claimed', 'says',
    'report', 'reports', 'today', 'said', 'according'
}

DEBUNK_TERMS = [
    "fact check", "fact-check", "debunk", "debunked", "debunking",
    "false claim", "claim is false", "misleading", "hoax", "fake news",
    "unfounded", "untrue", "no evidence", "satire", "parody",
    "fabricated", "fabrication", "disproven", "baseless", "denies rumor",
    "bogus", "scam", "conspiracy theory", "incorrectly claimed"
]

def extract_search_query(text: str) -> str:
    """
    Extracts a clean, concise search query from the headline or first sentence,
    stripping wire prefixes like 'WASHINGTON (Reuters) - ' or 'TEHRAN (AFP) - '.
    """
    cleaned = re.sub(r'^[A-Z\s,]+(?:\([A-Za-z\s]+\))?\s*[-—–]\s*', '', text.strip())
    sentences = re.split(r'[.\n!?]', cleaned)
    first_sentence = sentences[0].strip() if sentences else cleaned
    if len(first_sentence) > 180:
        words = first_sentence.split()
        first_sentence = " ".join(words[:25])
    return first_sentence if len(first_sentence) >= 10 else text.strip()[:180]

def get_real_time_evidence(query: str, api_key: str):
    """
    Queries Tavily AI Search to fetch real-time news evidence with titles, snippets, and URLs.
    """
    if not api_key or api_key == "USE_YOUR_API_KEY_HERE":
        return None, "API key not configured. Enter a valid Tavily API key in the sidebar."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=4)
        results = response.get('results', [])
        evidence_items = []
        for r in results:
            evidence_items.append({
                "title": r.get('title', 'News Source'),
                "content": r.get('content', ''),
                "url": r.get('url', '#')
            })
        return evidence_items, None
    except Exception as e:
        return None, f"Search Error: {str(e)}"

def analyze_evidence_stance(claim_text: str, evidence_items: list):
    """
    Performs stance and factual verification:
    1. Checks if search results contain debunking/refutation signals.
    2. Measures keyword/entity overlap to verify actual topical corroboration.
    Returns:
        stance: 'REFUTED', 'CORROBORATED', 'PARTIAL_MATCH', or 'NO_EVIDENCE'
    """
    if not evidence_items:
        return {
            'stance': 'NO_EVIDENCE',
            'is_debunked': False,
            'debunk_matches': [],
            'overlap_ratio': 0.0,
            'matched_keywords': [],
            'total_keywords': 0
        }

    # Extract keywords from the claim
    words = re.findall(r'[a-zA-Z0-9]+', claim_text.lower())
    keywords = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    if not keywords:
        keywords = words

    # Combine all evidence titles and snippets
    combined_evidence = " ".join([
        (item.get('title', '') + " " + item.get('content', '')).lower()
        for item in evidence_items
    ])

    # 1. Check for explicit debunking markers
    debunk_matches = [term for term in DEBUNK_TERMS if term in combined_evidence]
    is_debunked = len(debunk_matches) > 0

    # 2. Check keyword overlap for corroboration
    matched_keywords = [kw for kw in keywords if kw in combined_evidence]
    overlap_ratio = len(matched_keywords) / len(keywords) if keywords else 0.0

    # 3. Determine evidence stance
    if is_debunked:
        stance = 'REFUTED'
    elif overlap_ratio >= 0.45 and len(evidence_items) >= 1:
        stance = 'CORROBORATED'
    elif overlap_ratio >= 0.25:
        stance = 'PARTIAL_MATCH'
    else:
        stance = 'NO_EVIDENCE'

    return {
        'stance': stance,
        'is_debunked': is_debunked,
        'debunk_matches': debunk_matches,
        'overlap_ratio': round(overlap_ratio, 2),
        'matched_keywords': matched_keywords,
        'total_keywords': len(keywords)
    }

# ---------------------------------------------------------
# 5. Main UI Layout
# ---------------------------------------------------------
st.title("🛡️ Hybrid AI Fake News Detector")
st.markdown("### Cross-Referencing **Linguistic Fingerprint (BERT)** with **Real-Time Fact Verification (RAG)**")
st.caption(f"Engine status: {model_status}")
st.divider()

user_text = st.text_area(
    "Enter news headline or article to analyze:",
    height=140,
    placeholder="Paste a headline or article paragraph here to verify authenticity..."
)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_btn = st.button("🔍 Run Full Hybrid Analysis", use_container_width=True, type="primary")

if run_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing linguistic patterns and cross-referencing live databases..."):
            # ---------------------------------------------------------
            # PHASE 1: BERT Linguistic Analysis
            # ---------------------------------------------------------
            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()

            # Classify style into formal, sensationalist, or neutral/mixed
            if prob_fake >= 0.65:
                style_class = "SENSATIONAL"
                style_conf = prob_fake * 100
            elif prob_real >= 0.65:
                style_class = "FORMAL"
                style_conf = prob_real * 100
            else:
                style_class = "NEUTRAL"
                style_conf = max(prob_real, prob_fake) * 100

            # ---------------------------------------------------------
            # PHASE 2: Live RAG Search & Stance Verification
            # ---------------------------------------------------------
            search_query = extract_search_query(user_text)
            evidence_items, search_err = get_real_time_evidence(search_query, TAVILY_API_KEY)
            
            has_search = (evidence_items is not None and len(evidence_items) > 0)
            evidence_analysis = analyze_evidence_stance(user_text, evidence_items or [])

        # ---------------------------------------------------------
        # PHASE 3: Side-by-Side Analysis Display
        # ---------------------------------------------------------
        col_bert, col_rag = st.columns(2)

        # --- Column 1: BERT Analysis ---
        with col_bert:
            st.subheader("🤖 1. BERT Linguistic Style")
            
            if style_class == "FORMAL":
                st.success("📝 **STYLE: FORMAL / JOURNALISTIC**")
                st.write(f"Confidence: `{style_conf:.1f}%` match to objective news wire patterns.")
            elif style_class == "SENSATIONAL":
                st.error("🚨 **STYLE: SENSATIONALIST / CLICKBAIT**")
                st.write(f"Confidence: `{style_conf:.1f}%` match to emotive or manipulative language.")
            else:
                st.warning("⚖️ **STYLE: NEUTRAL / AMBIGUOUS**")
                st.write(f"Confidence: `{style_conf:.1f}%` balanced style (common in modern reporting).")

            st.caption("DistilBERT inspects syntax, emotional tone, and vocabulary fingerprints.")
            st.progress(float(prob_real), text=f"Formal: {prob_real*100:.1f}% | Sensationalist: {prob_fake*100:.1f}%")

        # --- Column 2: Live Detection ---
        with col_rag:
            st.subheader("🌐 2. Live Fact Verification")
            
            if not has_search:
                if search_err:
                    st.warning(f"⚠️ **STATUS: SEARCH UNAVAILABLE**")
                    st.caption(f"{search_err}")
                else:
                    st.warning("⚠️ **STATUS: NO LIVE MATCHES FOUND**")
                    st.write("Zero matching reports found in current news databases.")
            else:
                stance = evidence_analysis['stance']
                overlap_pct = int(evidence_analysis['overlap_ratio'] * 100)
                
                if stance == 'REFUTED':
                    st.error("🚨 **STATUS: ACTIVELY DEBUNKED**")
                    debunk_list = ", ".join(evidence_analysis['debunk_matches'][:3])
                    st.write(f"Fact-checks or refutation markers found: `{debunk_list}`.")
                elif stance == 'CORROBORATED':
                    st.success("✅ **STATUS: CORROBORATED BY NEWS**")
                    st.write(f"Found {len(evidence_items)} matching reports with `{overlap_pct}%` factual topic overlap.")
                elif stance == 'PARTIAL_MATCH':
                    st.info("ℹ️ **STATUS: PARTIAL TOPICAL MATCH**")
                    st.write(f"General topics found (`{overlap_pct}%` overlap), but claim lacks direct confirmation.")
                else:
                    st.warning("⚠️ **STATUS: NO DIRECT MATCH**")
                    st.write("Search returned general articles, but no confirmation for this specific claim.")

                with st.expander(f"View {len(evidence_items)} Live Sources & Snippets"):
                    st.caption(f"Search Query used: `{search_query}`")
                    for idx, item in enumerate(evidence_items, 1):
                        title = item.get('title', f"Source {idx}")
                        url = item.get('url', '#')
                        snippet = item.get('content', '')
                        st.markdown(f"**[{title}]({url})**")
                        st.write(f"_{snippet[:280]}..._")
                        st.divider()

        # ---------------------------------------------------------
        # PHASE 4: Robust Hybrid AI Verdict (Eliminating False Positives)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("🏁 Final AI Verdict")

        stance = evidence_analysis['stance']
        is_debunked = evidence_analysis['is_debunked']
        overlap_pct = int(evidence_analysis['overlap_ratio'] * 100)

        # Decision Matrix
        if not has_search:
            # When live search is unavailable, rely cautiously on style with clear disclaimers
            if style_class == "FORMAL":
                st.info("### ℹ️ VERDICT: LIKELY REAL (Linguistic Style Only)")
                st.markdown("**Reasoning:** The text follows formal journalistic style. However, live search is offline, so factual verification could not be confirmed.")
            elif style_class == "SENSATIONAL":
                st.warning("### ⚠️ VERDICT: SUSPICIOUS STYLE (Unverified)")
                st.markdown("**Reasoning:** Sensationalist/clickbait linguistic patterns detected. Connect Tavily API to verify facts against live databases.")
            else:
                st.warning("### ⚠️ VERDICT: INCONCLUSIVE")
                st.markdown("**Reasoning:** Neutral linguistic style with no live search verification available.")

        elif is_debunked:
            # Case 1: Actively debunked by fact-checkers (Catches viral fake news)
            st.error("### 🚨 VERDICT: FAKE NEWS (DEBUNKED)")
            debunk_terms_str = ", ".join(evidence_analysis['debunk_matches'][:3])
            st.markdown(f"**Reasoning:** Live fact-checking databases explicitly refute this claim (`{debunk_terms_str}`). Live evidence definitively disproves the assertion.")

        elif stance == 'CORROBORATED':
            # Case 2: Verified by live news outlets (Fixes false positives on modern real news!)
            st.success("### ✅ VERDICT: TRUE NEWS")
            if style_class == "SENSATIONAL":
                st.markdown(f"**Reasoning:** Corroborated by independent live news reports ({overlap_pct}% factual overlap). **Live factual evidence overrides BERT's stylistic suspicion**, resolving temporal drift.")
            else:
                st.markdown(f"**Reasoning:** Corroborated by independent live news reporting ({overlap_pct}% factual overlap) and consistent with formal journalistic reporting.")

        elif style_class == "SENSATIONAL" and stance in ('NO_EVIDENCE', 'PARTIAL_MATCH'):
            # Case 3: Sensationalist writing with zero confirmation (Catches rumors & fabricated stories)
            st.error("### 🚨 VERDICT: LIKELY FAKE NEWS")
            st.markdown("**Reasoning:** BERT detected sensationalist/manipulative patterns, and no credible news reports corroborate this claim in live databases.")

        elif style_class == "FORMAL" and stance in ('NO_EVIDENCE', 'PARTIAL_MATCH'):
            # Case 4: Formal tone, but no matching live news (Could be historical, local, or specialized)
            st.warning("### ⚠️ VERDICT: UNVERIFIED (Formal Style)")
            st.markdown("**Reasoning:** The article demonstrates formal journalistic structure, but no corroborating reports were found in current news databases. Please check primary historical or local archives.")

        else:
            # Case 5: Neutral style with weak search matches
            st.warning("### ⚠️ VERDICT: INCONCLUSIVE")
            st.markdown("**Reasoning:** Ambiguous linguistic cues and insufficient live corroboration. Independent manual verification is recommended.")

        # Verdict Breakdown Summary
        with st.expander("📊 View Decision Logic Breakdown"):
            st.write(f"- **BERT Formal Probability:** `{prob_real*100:.1f}%`")
            st.write(f"- **BERT Sensationalist Probability:** `{prob_fake*100:.1f}%`")
            st.write(f"- **Search Query:** `{search_query}`")
            st.write(f"- **Live Evidence Stance:** `{stance}`")
            st.write(f"- **Fact Check / Debunk Signals Detected:** `{is_debunked}`")
            st.write(f"- **Factual Overlap Ratio:** `{overlap_pct}%`")
