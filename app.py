"""
AI Find My Notes - Intelligent Academic Notes Retrieval
Using TF-IDF and Cosine Similarity

This app lets a student type a topic (e.g. "Explain heuristic search and
shortest path") and finds the PDF notes that are most related to that
topic, using classic text-similarity techniques (no AI models, no
internet, no API keys needed).

Just copy your PDF files into the dataset/ folder - no code changes
needed - and search. Clicking "View PDF" on a result opens that PDF
right inside the app, scrolled to the page where the topic was found.
"""

import os
import re
import base64
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# streamlit-pdf-viewer renders PDFs properly using pdf.js and supports
# jumping to a specific page. A plain <iframe src="data:..."> trick was
# tried first, but Chromium-based browsers (Chrome/Brave/Edge) silently
# block it once the PDF's base64 data gets long, showing a blank box -
# this component avoids that problem entirely.
try:
    from streamlit_pdf_viewer import pdf_viewer
    PDF_VIEWER_AVAILABLE = True
except ImportError:
    PDF_VIEWER_AVAILABLE = False

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_FOLDER = "dataset"
TOP_N_RESULTS = 5
PREVIEW_LENGTH = 300

# A score has to be strictly greater than 0 to be shown at all.
# TF-IDF + cosine similarity scores for a short query against a long
# document are naturally small (often 0.02-0.15) - that does NOT mean
# the match is wrong, it's just how the math works. We do NOT filter
# on a bigger cutoff like 0.05, because that was hiding real matches.
MIN_SCORE_TO_SHOW = 0.0

# Words shorter than this are ignored when hunting for the matching
# page (things like "a", "to", "is" would otherwise "win" a page match
# just by being common, which is exactly the wrong kind of guessing).
MIN_KEYWORD_LENGTH = 2

# ---------------------------------------------------------
# BACKGROUND IMAGE
# ---------------------------------------------------------
# Put any of the three images in an "assets" folder next to this file
# and set the filename below. The image is base64-embedded straight
# into the page's own CSS, so it works fully offline - no network
# request, same as the rest of the app.
ASSETS_FOLDER = "assets"
BACKGROUND_IMAGE_FILE = "bg-books-teal.png"   # swap to "bg-cozy-flatlay.png", "bg-bookshelf-wide.png" or "bg-open-book.png" any time


# ---------------------------------------------------------
# STEP 1: Find all PDF files in the dataset folder (recursively)
# ---------------------------------------------------------
def get_pdf_file_paths(folder_path):
    """Return the full path of every .pdf file inside folder_path (and subfolders)."""
    pdf_paths = []
    if not os.path.isdir(folder_path):
        return pdf_paths
    for root, _dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file_name))
    return pdf_paths


# ---------------------------------------------------------
# STEP 2: Extract raw text from a single PDF, PAGE BY PAGE
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF page by page and keeps each page's text separately
    (instead of merging everything into one blob). Keeping pages
    separate is what lets us later tell the user *which page number*
    their search topic was found on, and jump the PDF viewer there.

    Returns (page_texts, page_count, error_message).
    Some PDFs (scanned images, protected/corrupted files) may not have
    readable text - we handle that gracefully instead of crashing.
    """
    page_texts = []
    page_count = 0
    error_message = None
    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        for page in reader.pages:
            page_text = page.extract_text() or ""  # may return None
            page_texts.append(page_text)
    except Exception as error:
        error_message = str(error)
    return page_texts, page_count, error_message


# ---------------------------------------------------------
# STEP 3: Clean / preprocess text
# ---------------------------------------------------------
def clean_text(raw_text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = raw_text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------
# STEP 4: Load and index every PDF in the dataset folder
#          (cached so many PDFs are not re-read on every search)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Reading and indexing PDF notes...")
def load_pdf_notes(folder_path):
    """
    Scans the dataset folder, extracts + cleans text from every PDF,
    and returns:
      - file_names: names of PDFs with usable text
      - file_paths: full disk path for each PDF (needed to open/view it)
      - pages_texts: for each PDF, a list of per-page text
      - cleaned_texts: the WHOLE document's cleaned text (all pages
        joined) - this is what TF-IDF uses to score the document
      - diagnostics: per-file info (pages, chars extracted, error) -
        used to help debug scanning problems.
    """
    pdf_paths = get_pdf_file_paths(folder_path)

    file_names, file_paths, pages_texts, cleaned_texts = [], [], [], []
    diagnostics = []

    for pdf_path in pdf_paths:
        name = os.path.basename(pdf_path)
        page_texts, page_count, error = extract_text_from_pdf(pdf_path)
        whole_document_text = " ".join(page_texts)
        cleaned = clean_text(whole_document_text)

        diagnostics.append({
            "name": name,
            "pages": page_count,
            "chars_extracted": len(whole_document_text.strip()),
            "error": error,
        })

        if cleaned == "":
            continue  # no usable text - skip from search index

        file_names.append(name)
        file_paths.append(pdf_path)
        pages_texts.append(page_texts)
        cleaned_texts.append(cleaned)

    return file_names, file_paths, pages_texts, cleaned_texts, diagnostics


# ---------------------------------------------------------
# STEP 5: Find WHICH PAGE the query topic actually appears on most,
#          and build a short preview snippet from that page
# ---------------------------------------------------------
def find_matching_page(page_texts, query, preview_length=PREVIEW_LENGTH):
    """
    Scores EVERY page by how many times the query's meaningful words
    (stopwords like "the"/"a"/"of" removed - same stopword list the
    TF-IDF vectorizer uses, so page-matching and document-matching
    agree with each other) actually occur on that page, then returns
    the page with the HIGHEST score.

    This replaces the old "first page containing any query word"
    approach, which could point to the wrong page just because a
    common word happened to appear early in the document. This
    version is deterministic and defensible: the reported page number
    is always the page containing the most occurrences of the actual
    topic words, or None if none of those words appear anywhere at
    all (in which case we say so honestly instead of guessing page 1).

    Returns (page_number, snippet). page_number is None only when NO
    page contains ANY of the query's keywords.
    """
    query_words = [
        word for word in clean_text(query).split()
        if word not in ENGLISH_STOP_WORDS and len(word) > MIN_KEYWORD_LENGTH
    ]
    # If the query was made up entirely of stopwords/short words
    # (rare), fall back to using every word so we still have something
    # to search for.
    if not query_words:
        query_words = clean_text(query).split()

    best_page_index = None
    best_score = 0
    best_snippet = None

    for index, page_text in enumerate(page_texts):
        lower_page = page_text.lower()

        # Count total occurrences of every query keyword on this page.
        page_score = sum(lower_page.count(word) for word in query_words)

        if page_score > best_score:
            best_score = page_score
            best_page_index = index

            # Build the preview snippet around the first keyword that
            # actually occurs on this (best-so-far) page.
            best_snippet = None
            for word in query_words:
                position = lower_page.find(word)
                if position != -1:
                    start = max(0, position - 50)
                    end = start + preview_length
                    best_snippet = page_text[start:end].strip().replace("\n", " ")
                    break

    if best_page_index is None:
        # None of the query's real keywords appear on any page.
        # Do NOT guess page 1 - be upfront that page-level matching
        # failed even though the document as a whole scored a hit
        # (this can happen with related-vocabulary matches, e.g. the
        # doc talks about "Dijkstra" and TF-IDF still ranks it for a
        # query about "shortest path").
        return None, "(Topic words weren't found verbatim on any single page - document matched on related vocabulary.)"

    snippet = best_snippet if best_snippet else "(No preview available)"
    return best_page_index + 1, (snippet + "..." if best_snippet else snippet)


# ---------------------------------------------------------
# STEP 6: Search - TF-IDF + Cosine Similarity
# ---------------------------------------------------------
def search_notes(query, file_names, file_paths, pages_texts, cleaned_texts, top_n=TOP_N_RESULTS):
    """
    1. Combine all document texts + the query into one list.
    2. Fit TF-IDF on that combined list so query & documents share a vocabulary.
    3. Compute cosine similarity between the query vector and every document vector.
    4. Sort by similarity (highest first) and return the top N.
    """
    cleaned_query = clean_text(query)
    all_texts = cleaned_texts + [cleaned_query]

    # ngram_range=(1,2) lets 2-word phrases (e.g. "heuristic search") count too,
    # which usually improves matches for short queries.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vector = tfidf_matrix[-1]
    document_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)[0]

    results = list(zip(file_names, file_paths, pages_texts, similarity_scores))
    results.sort(key=lambda item: item[3], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------
def score_color(score):
    if score >= 0.3:
        return "#5a8f3d"   # deep olive-green - strong match
    elif score >= 0.1:
        return "#c9b25a"   # muted gold - moderate match
    else:
        return "#a37c4a"   # warm tan/brown - weak but present match


# A small hand-drawn SVG book, used as a header illustration. It's
# embedded directly in the page (no network request, no image file to
# manage) so the app stays fully offline, matching the "no internet
# needed" design of the rest of the tool.
BOOK_SVG = """
<svg width="92" height="76" viewBox="0 0 92 76" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="coverLeft" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#c3d69b"/>
      <stop offset="100%" stop-color="#6f9b50"/>
    </linearGradient>
    <linearGradient id="coverRight" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e2d488"/>
      <stop offset="100%" stop-color="#a3944f"/>
    </linearGradient>
  </defs>
  <path d="M46 14 C 36 6, 14 4, 4 8 L 4 66 C 14 62, 36 64, 46 72 Z" fill="url(#coverLeft)"/>
  <path d="M46 14 C 56 6, 78 4, 88 8 L 88 66 C 78 62, 56 64, 46 72 Z" fill="url(#coverRight)"/>
  <path d="M46 14 L 46 72" stroke="#2f3d27" stroke-width="1.5"/>
  <path d="M10 16 C 20 12, 34 13, 42 19" stroke="#eef1e7" stroke-width="1.4" fill="none" opacity="0.8"/>
  <path d="M10 26 C 20 22, 34 23, 42 29" stroke="#eef1e7" stroke-width="1.4" fill="none" opacity="0.6"/>
  <path d="M10 36 C 20 32, 34 33, 42 39" stroke="#eef1e7" stroke-width="1.4" fill="none" opacity="0.4"/>
  <path d="M82 16 C 72 12, 58 13, 50 19" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.8"/>
  <path d="M82 26 C 72 22, 58 23, 50 29" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.6"/>
  <path d="M82 36 C 72 32, 58 33, 50 39" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.4"/>
</svg>
"""


def get_background_css():
    """
    Reads BACKGROUND_IMAGE_FILE from ASSETS_FOLDER, base64-encodes it,
    and returns a CSS background rule that layers a dark, semi-opaque
    gradient OVER the photo (so the existing cream/orange text stays
    readable) and the photo itself underneath. If the file isn't there
    yet, falls back to the plain gradient background so the app never
    crashes just because the image hasn't been added.

    The path is resolved relative to THIS SCRIPT FILE, not the current
    working directory - if you run `streamlit run app.py` from a
    different folder (or launch it from an IDE), a plain "assets/..."
    path silently fails to be found because it gets looked up relative
    to wherever the terminal happens to be sitting, not next to app.py.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, ASSETS_FOLDER, BACKGROUND_IMAGE_FILE)
    plain_gradient = "linear-gradient(160deg, #16241a 0%, #0e1a12 45%, #080f0a 100%)"

    if not os.path.isfile(image_path):
        # Surface this loudly instead of silently falling back - a
        # missing/misnamed file is the #1 reason the background
        # "just doesn't show up" with no other clue why.
        st.warning(
            f"⚠️ Background image not found at: `{image_path}`\n\n"
            f"Make sure an `{ASSETS_FOLDER}/` folder sits right next to app.py "
            f"and contains a file named exactly `{BACKGROUND_IMAGE_FILE}`."
        )
        return f"background: {plain_gradient};"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    return f"""
        background-image:
            linear-gradient(160deg, rgba(10,20,13,0.90) 0%, rgba(6,12,8,0.93) 100%),
            url("data:image/{mime};base64,{encoded}");
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    """


def inject_custom_css():
    # NOTE: this stays a plain (non f-string) string because the CSS
    # below is full of literal { } braces - only the one background
    # rule is swapped in, via .replace(), to avoid having to escape
    # every curly brace in the whole stylesheet.
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            __BACKGROUND_CSS__
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .app-header {
            text-align: center;
            padding: 1.4rem 1rem 0.5rem 1rem;
        }
        .app-header .book-icon {
            filter: drop-shadow(0 6px 16px rgba(122, 158, 92, 0.45));
            margin-bottom: 0.4rem;
        }
        .app-header h1 {
            font-family: 'Fraunces', serif;
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            color: #d8c77a;
            text-shadow: 0 0 26px rgba(122, 158, 92, 0.35);
        }
        .app-header p {
            color: #c3cfb2;
            font-size: 1.08rem;
            font-weight: 500;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            border: 2px solid #3c4a34;
            background: #16211a;
            color: #eef1e7;
            padding: 0.8rem 1.1rem;
            font-size: 1rem;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #7ea172;
            box-shadow: 0 0 0 3px rgba(126,161,114,0.22);
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid #3c4a34;
            color: #e6ecd9;
            background: #16211a;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(126, 161, 114, 0.28);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #4c6b3a, #7ea172, #c3d69b);
            background-size: 200% 200%;
            border: none;
            color: #12200f;
            font-size: 1.02rem;
            padding: 0.55rem 0;
        }
        .stButton > button[kind="primary"]:hover {
            background-position: 100% 0;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 700;
            background: linear-gradient(90deg, #3f6b2a, #6fa050);
            border: none;
            color: #0c160a;
        }

        .result-card {
            background: linear-gradient(150deg, #182a19 0%, #0e1a10 100%);
            border: 1px solid #33422e;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        }
        .result-title {
            font-family: 'Fraunces', serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #eef1e7;
            margin-bottom: 0.5rem;
        }
        .score-badge {
            display: inline-block;
            padding: 0.2rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
            margin-right: 0.4rem;
            margin-top: 0.2rem;
        }
        .preview-text {
            color: #d6e0c8;
            font-size: 0.92rem;
            margin-top: 0.8rem;
            line-height: 1.55;
            background: rgba(126, 161, 114, 0.10);
            border-left: 3px solid #7ea172;
            padding: 0.6rem 0.9rem;
            border-radius: 0 10px 10px 0;
        }
        .stCaption, .st-emotion-cache-1629p8f { color: #c3cfb2 !important; }
        </style>
    """
    css = css.replace("__BACKGROUND_CSS__", get_background_css())
    st.markdown(css, unsafe_allow_html=True)


def render_result_card(rank, file_name, score, page_number):
    color = score_color(score)
    page_label = f"📄 Page {page_number}" if page_number else "📄 No exact page match"
    st.markdown(f"""
        <div class="result-card">
            <div class="result-title">🏆 {rank}. {file_name}</div>
            <span class="score-badge" style="background:{color}; color:#0e1117;">
                ⭐ Similarity: {score:.2f}
            </span>
            <span class="score-badge" style="background:#2d3a26; color:#dde6c9;">
                {page_label}
            </span>
        </div>
    """, unsafe_allow_html=True)


def render_pdf_viewer(file_path, page_number):
    """
    Renders the actual PDF inline using the streamlit-pdf-viewer
    component (built on pdf.js), scrolled straight to the page the
    topic was found on. This is far more reliable across browsers
    than embedding a base64 PDF inside a plain <iframe>, which
    Chromium browsers tend to block/show blank once the file is
    more than a couple of pages.
    """
    if not PDF_VIEWER_AVAILABLE:
        st.warning(
            "The inline PDF viewer needs one extra package. Run this in your "
            "terminal, then restart the app:\n\n`pip install streamlit-pdf-viewer`"
        )
        return

    open_at_page = page_number if page_number else 1
    pdf_viewer(
        input=file_path,
        width="100%",
        height=650,
        scroll_to_page=open_at_page,
    )


# ---------------------------------------------------------
# STREAMLIT USER INTERFACE
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Find My Notes", page_icon="📚", layout="centered")
    inject_custom_css()

    st.markdown(f"""
        <div class="app-header">
            <div class="book-icon">{BOOK_SVG}</div>
            <h1>AI Find My Notes</h1>
            <p>Search your academic notes using TF-IDF and Cosine Similarity ✨</p>
        </div>
    """, unsafe_allow_html=True)

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_names, file_paths, pages_texts, cleaned_texts, diagnostics = load_pdf_notes(DATASET_FOLDER)

    if len(diagnostics) == 0:
        st.info(
            f"No PDF files found. Please add your PDF notes to the "
            f"'{DATASET_FOLDER}/' folder and refresh the page."
        )
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"📁 {len(file_names)} of {len(diagnostics)} PDF(s) indexed and searchable.")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_button"):
            st.cache_data.clear()
            st.session_state.pop("search_results", None)
            st.rerun()

    # Diagnostics - helps confirm PDFs are actually being read correctly
    with st.expander("🔍 PDF scan details (open this if results look wrong)"):
        for item in diagnostics:
            if item["error"]:
                st.write(f"❌ **{item['name']}** — could not open: {item['error']}")
            elif item["chars_extracted"] == 0:
                st.write(
                    f"⚠️ **{item['name']}** — {item['pages']} page(s), "
                    f"but 0 characters of text extracted (likely a scanned/image-only PDF)."
                )
            else:
                st.write(
                    f"✅ **{item['name']}** — {item['pages']} page(s), "
                    f"{item['chars_extracted']} characters extracted."
                )

    query = st.text_input(
        "Enter a topic or question:",
        placeholder="e.g. Explain heuristic search and shortest path",
    )
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True, key="search_button")

    # IMPORTANT: Streamlit reruns the whole script on every click (even
    # clicking "View PDF" below). st.button() is only True for the exact
    # run it was clicked on, so we save the search results into
    # st.session_state - that way they stay visible even after you click
    # a "View PDF" or "Download" button on one of the results.
    if search_clicked:
        if query.strip() == "":
            st.warning("Please type a topic or question to search.")
            st.session_state.pop("search_results", None)
        elif len(file_names) == 0:
            st.info("No readable PDFs are available to search.")
        else:
            results = search_notes(query, file_names, file_paths, pages_texts, cleaned_texts)
            relevant_results = [r for r in results if r[3] > MIN_SCORE_TO_SHOW]
            st.session_state["search_results"] = relevant_results
            st.session_state["search_query"] = query

    # Render whatever the last search produced (persists across reruns)
    if "search_results" in st.session_state:
        relevant_results = st.session_state["search_results"]
        last_query = st.session_state.get("search_query", "")

        if not relevant_results:
            st.error(
                "No relevant notes were found. This usually means none of your "
                "PDFs contain any of the words in your query - try different "
                "keywords, or open 'PDF scan details' above to check the PDFs "
                "were read correctly."
            )
        else:
            st.subheader("Results")
            for rank, (file_name, file_path, page_texts, score) in enumerate(relevant_results, start=1):
                page_number, preview = find_matching_page(page_texts, last_query)

                render_result_card(rank, file_name, score, page_number)
                st.markdown(f'<div class="preview-text">{preview}</div>', unsafe_allow_html=True)

                view_key = f"view_open_{rank}_{file_name}"
                if view_key not in st.session_state:
                    st.session_state[view_key] = False

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    label = "🙈 Hide PDF" if st.session_state[view_key] else "📖 View PDF"
                    if st.button(label, key=f"toggle_{rank}_{file_name}", use_container_width=True):
                        st.session_state[view_key] = not st.session_state[view_key]
                        st.rerun()
                with btn_col2:
                    with open(file_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=f,
                            file_name=file_name,
                            mime="application/pdf",
                            key=f"download_{rank}_{file_name}",
                            use_container_width=True,
                        )

                if st.session_state[view_key]:
                    render_pdf_viewer(file_path, page_number)

                st.write("")  # small spacer between cards


if __name__ == "__main__":
    main()