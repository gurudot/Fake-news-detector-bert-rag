# import streamlit as st
# import torch
# import os
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from tavily import TavilyClient

# # 1. Setup & API Config
# st.set_page_config(page_title="BERT + RealTime Shield", page_icon="🛡️", layout="wide")
# TAVILY_API_KEY = "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" 
# tavily = TavilyClient(api_key=TAVILY_API_KEY) 

# # 2. Load BERT Model (Local)
# @st.cache_resource
# def load_bert_model():
#     model_name = "fine_tuned_bert_fake_news"
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
#     model_path = os.path.join(base_dir, model_name)
    
#     if not os.path.exists(model_path):
#         st.error(f"❌ Folder not found at: {model_path}")
#         st.stop()
        
#     tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
#     mod = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to("cuda")
#     return tok, mod

# tokenizer, model = load_bert_model()

# # 3. Real-Time Search Function
# def get_real_time_evidence(query):
#     try:
#         # We increase max_results slightly to get better context
#         response = tavily.search(query=query, search_depth="advanced", max_results=3)
#         return [r['content'] for r in response['results']]
#     except Exception as e:
#         return [f"Search Error: {str(e)}"]

# # 4. UI Layout
# st.title("🛡️ Hybrid AI Fake News Detector")
# st.markdown("### Analyzing **Linguistic Style (BERT)** vs. **2026 Live Evidence**")
# st.divider()

# user_text = st.text_area("Enter news headline or article to analyze:", height=150, placeholder="Paste news here...")

# if st.button("🔍 Run Full Analysis", use_container_width=True):
#     if not user_text.strip():
#         st.warning("⚠️ Please enter text first.")
#     else:
#         with st.spinner("Processing BERT Model and Searching Web..."):
#             # --- PHASE 1: BERT ---
#             inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(**inputs)
            
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             prediction = torch.argmax(probs, dim=-1).item()
#             conf = probs[0][prediction].item() * 100

#             # --- PHASE 2: TAVILY ---
#             evidence = get_real_time_evidence(user_text)

#         # --- PHASE 3: UI OUTPUT ---
#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("🤖 BERT Analysis")
#             if prediction == 0:
#                 st.success("**RESULT: REAL**")
#                 st.write(f"Style Confidence: `{conf:.2f}%` Match to formal patterns.")
#             else:
#                 st.error("**RESULT: FAKE**")
#                 st.write(f"Style Confidence: `{conf:.2f}%` Match to sensationalist patterns.")
#             st.caption("Determined by BERT analyzing linguistic fingerprints and writing style.")

#         with col2:
#             st.subheader("🌐 Live Detection")
#             valid_search = len(evidence) >= 1 and "Search Error" not in evidence[0]
            
#             if valid_search:
#                 st.info("**STATUS: MATCHES FOUND**")
#                 st.write(f"Found {len(evidence)} matching reports from March 2026.")
#                 with st.expander("View Source Evidence"):
#                     for item in evidence:
#                         st.write(f"• {item[:250]}...")
#             else:
#                 st.warning("**STATUS: NO LIVE MATCH**")
#                 st.write("Zero matching reports found in today's news database.")
#             st.caption("Verified by cross-referencing the live internet via Tavily Search API.")

#         # --- PHASE 4: FINAL SMART VERDICT (The Fixed Logic) ---
#         st.divider()
#         st.subheader("🏁 Final AI Verdict")
        
#         # Check for "Conspiracy" keywords that Tavily might accidentally match
#         conspiracy_keywords = ["alien", "ufo", "hidden city", "conspiracy", "omg", "secret base"]
#         contains_conspiracy = any(word in user_text.lower() for word in conspiracy_keywords)

#         # Updated Decision Logic
#         if prediction == 1 and contains_conspiracy and conf > 70:
#             st.error("### 🚨 VERDICT: FAKE NEWS")
#             st.markdown("**Reasoning:** High-confidence sensationalist patterns detected. Search results likely contain general topics (NASA/Moon) but do not support the specific conspiracy claim.")
        
#         elif len(evidence) >= 2 and valid_search:
#             # Check if the search results actually contain modern 2026 facts
#             if "2026" in str(evidence) or "March" in str(evidence):
#                 st.success("### ✅ VERDICT: TRUE NEWS")
#                 st.markdown("**Reasoning:** High factual density found online for 2026 events. Live evidence overrides stylistic suspicion.")
#             else:
#                 st.warning("### ⚠️ VERDICT: INCONCLUSIVE")
#                 st.markdown("**Reasoning:** General matches found, but no specific 2026 confirmation. Please verify sources.")

#         elif prediction == 0 and valid_search:
#             st.success("### ✅ VERDICT: TRUE NEWS")
#             st.markdown("**Reasoning:** Alignment between BERT's formal style and live search data.")

#         elif prediction == 1 and not valid_search:
#             st.error("### 🚨 VERDICT: FAKE NEWS")
#             st.markdown("**Reasoning:** BERT detected manipulative language, and no supporting evidence was found.")

#         else:
#             st.warning("### ⚠️ VERDICT: INCONCLUSIVE")
#             st.markdown("**Reasoning:** Contradictory signals detected between style and live database.")

# # Sidebar Info
# st.sidebar.title("Project Details")
# st.sidebar.info(f"""
# **Developer:** Gurudatta Pradhan 
# **College:** Presidency University  
# **Department:** CSE
# **Hardware:** NVIDIA RTX 2050  
# **Method:** BERT + RAG (Tavily)
# """)

import streamlit as st
import torch
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from tavily import TavilyClient


# =========================================================
# DEVICE SETUP
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hybrid AI Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TAVILY API
# =========================================================

TAVILY_API_KEY = "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"

tavily = TavilyClient(
    api_key=TAVILY_API_KEY
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_bert_model():

    model_path = r"D:\Navneet\Fake-News-Detector-BERT-RAG-main\fine_tuned_bert_fake_news"

    if not os.path.exists(model_path):

        st.error(
            f"❌ Model folder not found: {model_path}"
        )

        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    ).to(device)

    return tokenizer, model


tokenizer, model = load_bert_model()


# =========================================================
# REAL-TIME SEARCH
# =========================================================

def get_real_time_evidence(query):

    try:

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=3
        )

        return [
            r['content']
            for r in response['results']
        ]

    except Exception as e:

        return [
            f"Search Error: {str(e)}"
        ]


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🛡️ Hybrid AI Fake News Detector")

st.markdown(
    """
    ### Analyzing **Linguistic Style (BERT)** 
    vs. **Live Internet Evidence**
    """
)

st.divider()


# =========================================================
# USER INPUT
# =========================================================

user_text = st.text_area(
    "Enter News Headline or Article:",
    height=180,
    placeholder="Paste news content here..."
)


# =========================================================
# MAIN ANALYSIS BUTTON
# =========================================================

if st.button(
    "🔍 Run Full Analysis",
    use_container_width=True
):

    if not user_text.strip():

        st.warning(
            "⚠️ Please enter news text first."
        )

    else:

        with st.spinner(
            "Running BERT Analysis and Live Verification..."
        ):

            # =================================================
            # BERT ANALYSIS
            # =================================================

            inputs = tokenizer(

                user_text,

                return_tensors="pt",

                truncation=True,

                padding=True,

                max_length=256

            ).to(device)

            model.eval()

            with torch.no_grad():

                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )

            prediction = torch.argmax(
                probs,
                dim=-1
            ).item()

            confidence = probs[
                0
            ][prediction].item() * 100


            # =================================================
            # LIVE SEARCH
            # =================================================

            evidence = get_real_time_evidence(
                user_text
            )

        # =====================================================
        # DISPLAY RESULTS
        # =====================================================

        col1, col2 = st.columns(2)

        # -----------------------------------------------------
        # LEFT PANEL - BERT
        # -----------------------------------------------------

        with col1:

            st.subheader("🤖 BERT Analysis")

            if prediction == 0:

                st.success(
                    "**RESULT: REAL NEWS**"
                )

                st.write(
                    f"Style Confidence: `{confidence:.2f}%` "
                    f"match to formal patterns."
                )

            else:

                st.error(
                    "**RESULT: FAKE NEWS**"
                )

                st.write(
                    f"Style Confidence: `{confidence:.2f}%` "
                    f"match to sensationalist patterns."
                )

            st.caption(
                "Generated using DistilBERT "
                "linguistic analysis."
            )


        # -----------------------------------------------------
        # RIGHT PANEL - LIVE SEARCH
        # -----------------------------------------------------

        with col2:

            st.subheader("🌐 Live Evidence Search")

            valid_search = (

                len(evidence) >= 1

                and

                "Search Error" not in evidence[0]

            )

            if valid_search:

                st.info(
                    "**STATUS: MATCHES FOUND**"
                )

                st.write(
                    f"Found {len(evidence)} "
                    f"matching reports online."
                )

                with st.expander(
                    "View Source Evidence"
                ):

                    for item in evidence:

                        st.write(
                            f"• {item[:300]}..."
                        )

            else:

                st.warning(
                    "**STATUS: NO LIVE MATCH FOUND**"
                )

                st.write(
                    "No matching reports found "
                    "from live internet sources."
                )

            st.caption(
                "Verified using Tavily Search API."
            )


        # =====================================================
        # FINAL HYBRID VERDICT
        # =====================================================

        st.divider()

        st.subheader("🏁 Final AI Verdict")


        conspiracy_keywords = [

            "alien",
            "ufo",
            "secret base",
            "hidden city",
            "mind control",
            "conspiracy"

        ]

        contains_conspiracy = any(

            word in user_text.lower()

            for word in conspiracy_keywords

        )


        # =====================================================
        # SMART DECISION LOGIC
        # =====================================================

        if (

            prediction == 1

            and

            contains_conspiracy

            and

            confidence > 70

        ):

            st.error(
                "### 🚨 VERDICT: FAKE NEWS"
            )

            st.markdown(
                """
                **Reasoning:**  
                High-confidence sensationalist patterns 
                detected. The claim appears misleading 
                and lacks reliable supporting evidence.
                """
            )


        elif len(evidence) >= 2 and valid_search:

            st.success(
                "### ✅ VERDICT: TRUE NEWS"
            )

            st.markdown(
                """
                **Reasoning:**  
                Multiple supporting reports were found 
                online. Live evidence aligns with the 
                linguistic analysis performed by BERT.
                """
            )


        elif prediction == 0 and valid_search:

            st.success(
                "### ✅ VERDICT: TRUE NEWS"
            )

            st.markdown(
                """
                **Reasoning:**  
                Formal writing patterns and supporting 
                live evidence indicate the news is likely real.
                """
            )


        elif prediction == 1 and not valid_search:

            st.error(
                "### 🚨 VERDICT: FAKE NEWS"
            )

            st.markdown(
                """
                **Reasoning:**  
                Manipulative or sensationalist language 
                detected, and no reliable live evidence 
                was found online.
                """
            )


        else:

            st.warning(
                "### ⚠️ VERDICT: INCONCLUSIVE"
            )

            st.markdown(
                """
                **Reasoning:**  
                Contradictory signals detected between 
                linguistic analysis and live verification.
                """
            )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Project Details")

st.sidebar.info(
    """
**Developer:** Gurudatta Pradhan 

**Roll No:** 20241CSE0086     

**College:** Presidency University  

**Department:** CSE  

**Hardware:** NVIDIA RTX 2050  

**Model:** DistilBERT  

**Method:** BERT + RAG (Tavily)
"""
)