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
import time
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

# How many past searches to remember as clickable "recent search" chips.
MAX_RECENT_QUERIES = 5

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
# STEP 5: Highlight the query's keywords inside a preview snippet
# ---------------------------------------------------------
def highlight_keywords(snippet, keywords):
    """
    Wraps every occurrence of any keyword (case-insensitive) in an
    HTML <mark> tag, so the matched words stand out visually inside
    the preview text instead of the student having to hunt for them.
    """
    if not keywords:
        return snippet
    pattern = re.compile(
        "(" + "|".join(re.escape(word) for word in keywords) + ")",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", snippet)


# ---------------------------------------------------------
# STEP 6: Find WHICH PAGE the query topic actually appears on most,
#          and build a short, keyword-highlighted preview snippet
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

    Returns (page_number, snippet_html). The snippet has the matched
    keywords wrapped in <mark> tags for visual highlighting.
    page_number is None only when NO page contains ANY of the query's
    keywords.
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
    if best_snippet:
        snippet = highlight_keywords(snippet, query_words) + "..."
    return best_page_index + 1, snippet


# ---------------------------------------------------------
# STEP 7: Search - TF-IDF + Cosine Similarity
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
# STEP 8: Run a search end-to-end (used by both the Search button
#          and the recent-search chips), with a brief loading state
#          and toast-style feedback
# ---------------------------------------------------------
def run_search(query, file_names, file_paths, pages_texts, cleaned_texts, loading_placeholder):
    """
    Shared search logic so both the main Search button and the
    "recent search" chips behave identically: validate the query,
    show a brief skeleton/shimmer loading placeholder, run the actual
    TF-IDF search, store results in session_state (so they survive
    later reruns from View/Download button clicks), and update the
    "recent searches" chip list.
    """
    if query.strip() == "":
        st.warning("Please type a topic or question to search.")
        st.session_state.pop("search_results", None)
        return
    if len(file_names) == 0:
        st.info("No readable PDFs are available to search.")
        return

    # Show a shimmering placeholder while the search "processes" - on a
    # small note collection this search is nearly instant, so a short,
    # clearly-labeled pause makes the loading state visible instead of
    # flashing by unnoticed. Remove the time.sleep line for zero delay.
    loading_placeholder.markdown(SKELETON_HTML, unsafe_allow_html=True)
    time.sleep(0.35)

    results = search_notes(query, file_names, file_paths, pages_texts, cleaned_texts)
    relevant_results = [r for r in results if r[3] > MIN_SCORE_TO_SHOW]
    loading_placeholder.empty()

    st.session_state["search_results"] = relevant_results
    st.session_state["search_query"] = query
    
    if relevant_results:
        recent = st.session_state.get("recent_queries", [])
        if query in recent:
            recent.remove(query)
        recent.insert(0, query)
        st.session_state["recent_queries"] = recent[:MAX_RECENT_QUERIES]


def mark_reindexed():
    """Records the time the PDF index was last (re)built, for the stats strip."""
    st.session_state["last_indexed_at"] = time.strftime("%I:%M %p")


# ---------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------
def score_color(score):
    if score >= 0.3:
        return "#d8a24a"   # warm gold - strong match
    elif score >= 0.1:
        return "#c97b8a"   # dusty rose - moderate match
    else:
        return "#7a4a3a"   # muted brown - weak but present match


# A small hand-drawn SVG book, used as a header illustration. It's
# embedded directly in the page (no network request, no image file to
# manage) so the app stays fully offline, matching the "no internet
# needed" design of the rest of the tool.
BOOK_SVG = """
<svg width="92" height="76" viewBox="0 0 92 76" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="coverLeft" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#c97b8a"/>
      <stop offset="100%" stop-color="#9a3a4a"/>
    </linearGradient>
    <linearGradient id="coverRight" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e2d488"/>
      <stop offset="100%" stop-color="#a3944f"/>
    </linearGradient>
  </defs>
  <path d="M46 14 C 36 6, 14 4, 4 8 L 4 66 C 14 62, 36 64, 46 72 Z" fill="url(#coverLeft)"/>
  <path d="M46 14 C 56 6, 78 4, 88 8 L 88 66 C 78 62, 56 64, 46 72 Z" fill="url(#coverRight)"/>
  <path d="M46 14 L 46 72" stroke="#3a1a1a" stroke-width="1.5"/>
  <path d="M10 16 C 20 12, 34 13, 42 19" stroke="#f5e6e6" stroke-width="1.4" fill="none" opacity="0.8"/>
  <path d="M10 26 C 20 22, 34 23, 42 29" stroke="#f5e6e6" stroke-width="1.4" fill="none" opacity="0.6"/>
  <path d="M10 36 C 20 32, 34 33, 42 39" stroke="#f5e6e6" stroke-width="1.4" fill="none" opacity="0.4"/>
  <path d="M82 16 C 72 12, 58 13, 50 19" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.8"/>
  <path d="M82 26 C 72 22, 58 23, 50 29" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.6"/>
  <path d="M82 36 C 72 32, 58 33, 50 39" stroke="#fbf6df" stroke-width="1.4" fill="none" opacity="0.4"/>
</svg>
"""

# One shimmering placeholder "card" shown briefly while a search runs.
SKELETON_HTML = """
<div class="skeleton-card"></div>
<div class="skeleton-card"></div>
"""


def get_background_image_data():
    """
    Reads BACKGROUND_IMAGE_FILE from ASSETS_FOLDER and returns its
    base64-encoded bytes plus MIME type, or None if the file isn't
    there. The path is resolved relative to THIS SCRIPT FILE, not the
    current working directory - if you run `streamlit run app.py` from
    a different folder (or launch it from an IDE), a plain "assets/..."
    path silently fails to be found because it gets looked up relative
    to wherever the terminal happens to be sitting, not next to app.py.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, ASSETS_FOLDER, BACKGROUND_IMAGE_FILE)

    if not os.path.isfile(image_path):
        # Surface this loudly instead of silently falling back - a
        # missing/misnamed file is the #1 reason the background
        # "just doesn't show up" with no other clue why.
        st.warning(
            f"⚠️ Background image not found at: `{image_path}`\n\n"
            f"Make sure an `{ASSETS_FOLDER}/` folder sits right next to app.py "
            f"and contains a file named exactly `{BACKGROUND_IMAGE_FILE}`."
        )
        return None

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return mime, encoded


def get_background_css():
    """
    Builds the full background layer as CSS. When the photo is found,
    it's placed on a fixed, full-viewport ::before pseudo-element
    behind everything, with a slow "Ken Burns" zoom/pan animation -
    the actual app containers are made transparent so this layer shows
    through. A dark gradient ::after layer sits on top of the photo
    (but still behind the real content) so existing text stays
    readable. If the photo is missing, falls back to a plain static
    gradient with no animation so the app never crashes.
    """
    plain_gradient = "linear-gradient(160deg, #2b0f16 0%, #1a0a0e 45%, #0d0608 100%)"
    image_data = get_background_image_data()

    if image_data is None:
        return f"""
            html, body {{
                background-color: #1a0a0e !important;
            }}

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {{
                background: {plain_gradient} !important;
            }}
        """

    mime, encoded = image_data
    return f"""
        html, body {{
            background-color: #1a0a0e !important;
        }}

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background: transparent !important;
        }}

        body::before {{
            content: "";
            position: fixed;
            inset: -3%;
            background-image: url("data:image/{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            z-index: -2;
            animation: bgKenBurns 36s ease-in-out infinite alternate;
        }}

        body::after {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(160deg, rgba(26,10,14,0.90) 0%, rgba(13,6,8,0.93) 100%);
            z-index: -1;
        }}

        @keyframes bgKenBurns {{
            0%   {{ transform: scale(1) translate(0, 0); }}
            50%  {{ transform: scale(1.10) translate(-1%, 1%); }}
            100% {{ transform: scale(1.16) translate(1%, -1%); }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            body::before {{ animation: none; }}
        }}
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

        __BACKGROUND_CSS__

        [data-testid="stHeader"] {
            background: transparent;
        }

        /* Sidebar has its own separate background container that the
           main __BACKGROUND_CSS__ rules never touch - without this it
           stays Streamlit's default dark slate gray regardless of the
           rest of the theme. Both testids are targeted since this
           differs slightly across Streamlit versions. */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #200a0f 0%, #140809 100%) !important;
            border-right: 1px solid #4a1f28;
        }
        section[data-testid="stSidebar"] h3 {
            font-family: 'Fraunces', serif;
            color: #d8c77a;
        }

        /* Expanders ("PDF scan details", "Manage PDFs") also default to
           Streamlit's own gray box styling until themed explicitly. */
        [data-testid="stExpander"] {
            background: #241019;
            border: 1px solid #4a1f28 !important;
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary {
            color: #f5e6e6;
        }
        [data-testid="stExpander"] summary:hover {
            color: #d8c77a;
        }

        .app-header {
            text-align: center;
            padding: 1.4rem 1rem 0.5rem 1rem;
        }
        .app-header .book-icon {
            filter: drop-shadow(0 6px 16px rgba(154, 58, 74, 0.45));
            margin-bottom: 0.4rem;
            animation: iconGlow 4s ease-in-out infinite;
        }
        @keyframes iconGlow {
            0%, 100% { filter: drop-shadow(0 6px 16px rgba(216, 162, 74, 0.45)); }
            50%      { filter: drop-shadow(0 6px 20px rgba(154, 58, 74, 0.55)); }
        }
        .app-header h1 {
            font-family: 'Fraunces', serif;
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            background: linear-gradient(90deg, #c97b8a, #d8c77a, #9a3a4a, #d8c77a, #c97b8a);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: titleGradient 8s ease-in-out infinite;
            text-shadow: 0 0 30px rgba(154, 58, 74, 0.25);
        }
        @keyframes titleGradient {
            0%, 100% { background-position: 0% 50%; }
            50%      { background-position: 100% 50%; }
        }
        .app-header p {
            color: #d9b7bd;
            font-size: 1.08rem;
            font-weight: 500;
        }

        /* Stats strip: small pill badges showing PDF count, page count,
           and last-indexed time under the header. */
        .stats-strip {
            display: flex;
            gap: 0.6rem;
            justify-content: center;
            flex-wrap: wrap;
            margin: 0.4rem 0 1.3rem;
        }
        .stat-pill {
            background: #241019;
            border: 1px solid #522530;
            border-radius: 999px;
            padding: 0.35rem 1rem;
            font-size: 0.82rem;
            color: #d9b7bd;
        }
        .stat-pill b { color: #c97b8a; }

        div[data-testid="stTextInput"] input {
            border-radius: 12px;
            border: 2px solid #522530;
            background: #241019;
            color: #f5e6e6;
            padding: 0.8rem 1.1rem;
            font-size: 1rem;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #9a3a4a;
            box-shadow: 0 0 0 3px rgba(154, 58, 74, 0.20);
        }

        /* Base button styling. Streamlit puts a plain st.button() inside
           a div[data-testid="stButton"] wrapper, but a
           st.form_submit_button() (used for the Search button, since
           it lives inside st.form) gets wrapped in a DIFFERENT
           container, div[data-testid="stFormSubmitButton"] - a rule
           that only says ".stButton > button" silently misses it. Both
           wrappers are targeted here so every button gets styled. */
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid #522530;
            color: #f0dede;
            background: #241019;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(154, 58, 74, 0.25);
        }
        /* Primary buttons: newer Streamlit versions render the primary
           style as data-testid="stBaseButton-primary" instead of the
           older kind="primary" HTML attribute - both are matched here
           so this keeps working across Streamlit versions. */
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
        div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(90deg, #3a1420, #6b2431, #4a1620);
            background-size: 200% 200%;
            border: none;
            color: #e8c98a;
            font-size: 1.02rem;
            padding: 0.55rem 0;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
        div[data-testid="stButton"] > button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primary"]:hover {
            background-position: 100% 0;
        }

        /* NOTE: an earlier version tried a ".chip-row div[...] > button"
           rule here, wrapping the chip buttons in an st.markdown() div.
           That never actually worked - st.markdown() inserts a sibling
           element, not a real parent, so Streamlit's buttons never end
           up nested inside it. Removed; the recent-search chips just
           use the standard button styling above, which already looks
           right now that it's actually applying (see the fix below).
           The narrow st.columns() width is what gives them their
           compact, chip-like look. */

        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 700;
            background: linear-gradient(90deg, #8a5a1f, #d8a24a);
            border: none;
            color: #2a1608;
        }

        /* Shimmering placeholder shown briefly while a search runs. */
        .skeleton-card {
            height: 84px;
            border-radius: 16px;
            margin-bottom: 0.8rem;
            border: 1px solid #4a1f28;
            background: linear-gradient(100deg, #241019 30%, #22331f 50%, #241019 70%);
            background-size: 200% 100%;
            animation: shimmer 1.1s linear infinite;
        }
        @keyframes shimmer {
            0%   { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .result-card {
            background: linear-gradient(150deg, #2e131a 0%, #170a0d 100%);
            border: 1px solid #4a1f28;
            border-left: 3px solid #7a2e3a;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            opacity: 0;
            transform: translateY(10px);
            animation: cardFadeIn 0.45s ease forwards;
        }
        @keyframes cardFadeIn {
            to { opacity: 1; transform: translateY(0); }
        }
        .result-title {
            font-family: 'Fraunces', serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #f5e6e6;
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
            color: #e8cdd2;
            font-size: 0.92rem;
            margin-top: 0.8rem;
            line-height: 1.55;
            background: rgba(154, 58, 74, 0.08);
            border-left: 3px solid #9a3a4a;
            padding: 0.6rem 0.9rem;
            border-radius: 0 10px 10px 0;
        }
        .preview-text mark {
            background: #d8c77a;
            color: #1a1400;
            padding: 0 0.2em;
            border-radius: 3px;
            font-weight: 700;
        }
        .stCaption, .st-emotion-cache-1629p8f { color: #d9b7bd !important; }

        /* Streamlit's built-in st.error/warning/info/success boxes ship
           with red/blue/orange colors baked in via inline styles, which
           clash with an olive theme. Overriding with !important on the
           stable data-testid wrapper (rather than the unstable, version-
           specific emotion-cache class names) recolors all four to a
           single consistent olive-toned card so nothing reads as an
           off-theme red/blue alert. */
        [data-testid="stAlert"] {
            background-color: #2e1219 !important;
            border: 1px solid #522530 !important;
            border-left: 4px solid #9a3a4a !important;
            border-radius: 10px !important;
            color: #f5e6e6 !important;
        }
        [data-testid="stAlert"] * {
            color: #f5e6e6 !important;
        }
        [data-testid="stAlert"] svg {
            fill: #c97b8a !important;
        }
        </style>
    """
    css = css.replace("__BACKGROUND_CSS__", get_background_css())
    st.markdown(css, unsafe_allow_html=True)


def render_result_card(rank, file_name, score, page_number, animation_delay):
    color = score_color(score)
    page_label = f"📄 Page {page_number}" if page_number else "📄 No exact page match"
    st.markdown(f"""
        <div class="result-card" style="animation-delay:{animation_delay}s;">
            <div class="result-title">🏆 {rank}. {file_name}</div>
            <span class="score-badge" style="background:{color}; color:#1a0e0e;">
                ⭐ Similarity: {score:.2f}
            </span>
            <span class="score-badge" style="background:#3a1a22; color:#f0d8dc;">
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
# SIDEBAR: diagnostics + upload/delete management
# ---------------------------------------------------------
def render_sidebar(file_names, file_paths, diagnostics):
    """
    Everything that's about MANAGING the note collection (rather than
    searching it) lives in the sidebar, so the main area stays focused
    on search + results.
    """
    with st.sidebar:
        st.markdown("### 📚 Notes Library")

        if st.button("🔄 Refresh index", use_container_width=True, key="refresh_button"):
            st.cache_data.clear()
            st.session_state.pop("search_results", None)
            mark_reindexed()
            st.rerun()

        with st.expander("🔍 PDF scan details"):
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

        with st.expander("🛠️ Manage PDFs (add or remove)"):
            tab_add, tab_remove = st.tabs(["➕ Add", "🗑️ Remove"])

            with tab_add:
                st.caption(
                    "Uploaded files are added to dataset/ and become searchable "
                    "immediately. (On Streamlit Cloud, uploads live only until the "
                    "app restarts or sleeps - for permanent notes, add them to "
                    "dataset/ in your GitHub repo instead.)"
                )
                uploaded_files = st.file_uploader(
                    "Choose one or more PDF files",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="pdf_uploader",
                )
                if uploaded_files:
                    if st.button("📥 Add to my notes", use_container_width=True, key="add_uploaded_pdfs"):
                        added, skipped = [], []
                        for uploaded_file in uploaded_files:
                            # os.path.basename strips any folder path a browser might send,
                            # so this can never write outside the dataset folder.
                            safe_name = os.path.basename(uploaded_file.name)
                            if not safe_name.lower().endswith(".pdf"):
                                continue
                            dest_path = os.path.join(DATASET_FOLDER, safe_name)
                            if os.path.exists(dest_path):
                                skipped.append(safe_name)
                                continue
                            with open(dest_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            added.append(safe_name)

                        # The index was built with @st.cache_data, so it has to be
                        # cleared explicitly - otherwise the newly-saved files would
                        # sit on disk but stay invisible to search until the cache
                        # expired on its own.
                        st.cache_data.clear()
                        st.session_state.pop("search_results", None)
                        mark_reindexed()

                        # st.toast() shows a small, auto-dismissing notification
                        # (rather than a static box that sits on the page) - it's
                        # specifically designed to survive the st.rerun() below.
                        if added:
                            st.toast(f"Added {len(added)} PDF(s): {', '.join(added)}", icon="✅")
                        if skipped:
                            st.toast(f"Skipped (already exists): {', '.join(skipped)}", icon="ℹ️")
                        st.rerun()

            with tab_remove:
                if len(file_names) == 0:
                    st.caption("No PDFs are currently indexed.")
                else:
                    st.caption("Removing a file deletes it from the dataset/ folder on disk.")
                    for name, path in zip(file_names, file_paths):
                        col_name, col_delete = st.columns([4, 1])
                        with col_name:
                            st.write(f"📄 {name}")
                        with col_delete:
                            # Delete happens in two clicks: first click arms a
                            # confirmation for THIS specific file only, so a stray
                            # click can't silently wipe out a student's notes.
                            confirm_key = f"confirm_delete_{name}"
                            if st.session_state.get(confirm_key):
                                if st.button("✅", key=f"confirm_btn_{name}", use_container_width=True, help=f"Confirm delete {name}"):
                                    try:
                                        os.remove(path)
                                        st.cache_data.clear()
                                        st.session_state.pop("search_results", None)
                                        st.session_state.pop(confirm_key, None)
                                        mark_reindexed()
                                        st.toast(f"Removed {name}.", icon="🗑️")
                                        st.rerun()
                                    except Exception as error:
                                        st.error(f"Could not remove {name}: {error}")
                            else:
                                if st.button("🗑️", key=f"delete_btn_{name}", use_container_width=True, help=f"Remove {name}"):
                                    st.session_state[confirm_key] = True
                                    st.rerun()


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

    st.session_state.setdefault("last_indexed_at", time.strftime("%I:%M %p"))
    st.session_state.setdefault("recent_queries", [])

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    file_names, file_paths, pages_texts, cleaned_texts, diagnostics = load_pdf_notes(DATASET_FOLDER)

    if len(diagnostics) == 0:
        st.info(
            f"No PDF files found. Please add your PDF notes to the "
            f"'{DATASET_FOLDER}/' folder and refresh the page."
        )
        render_sidebar(file_names, file_paths, diagnostics)
        return

    # Stats strip - quick at-a-glance numbers under the header
    total_pages = sum(item["pages"] for item in diagnostics)
    st.markdown(f"""
        <div class="stats-strip">
            <div class="stat-pill">📁 <b>{len(file_names)}</b> PDFs indexed</div>
            <div class="stat-pill">📄 <b>{total_pages}</b> pages scanned</div>
            <div class="stat-pill">🕒 Updated <b>{st.session_state['last_indexed_at']}</b></div>
        </div>
    """, unsafe_allow_html=True)

    render_sidebar(file_names, file_paths, diagnostics)

    # Wrapped in st.form so pressing Enter inside the text box submits the
    # search - a plain st.text_input + st.button outside a form only
    # responds to the button click, not the Enter key.
    with st.form(key="search_form"):
        query = st.text_input(
            "Enter a topic or question:",
            key="query_text",
            placeholder="e.g. Explain heuristic search and shortest path",
        )
        search_clicked = st.form_submit_button(
            "🔎 Search", type="primary", use_container_width=True
        )

    # Recent-search chips - click one to instantly re-run that search.
    # These are plain st.button() calls (not form_submit_button), which
    # is why they have to live outside the st.form block above - a form
    # can only contain one submit button.
    recent_queries = st.session_state.get("recent_queries", [])
    chip_clicked_query = None
    if recent_queries:
        st.caption("Recent searches:")
        chip_columns = st.columns(len(recent_queries))
        for column, past_query in zip(chip_columns, recent_queries):
            with column:
                if st.button(past_query, key=f"chip_{past_query}"):
                    chip_clicked_query = past_query

    # A single placeholder used for the brief shimmer/skeleton loading
    # effect right before results are drawn below it.
    loading_placeholder = st.empty()

    # IMPORTANT: Streamlit reruns the whole script on every click (even
    # clicking "View PDF" below). st.button() is only True for the exact
    # run it was clicked on, so search results are stored in
    # st.session_state - that way they stay visible even after clicking
    # a "View PDF" or "Download" button on one of the results.
    if search_clicked:
        run_search(query, file_names, file_paths, pages_texts, cleaned_texts, loading_placeholder)
    elif chip_clicked_query is not None:
        run_search(chip_clicked_query, file_names, file_paths, pages_texts, cleaned_texts, loading_placeholder)

    # Render whatever the last search produced (persists across reruns)
    if "search_results" in st.session_state:
        relevant_results = st.session_state["search_results"]

        if not relevant_results:
            st.error(
                "No relevant notes were found. This usually means none of your "
                "PDFs contain any of the words in your query - try different "
                "keywords, or open 'PDF scan details' in the sidebar to check the "
                "PDFs were read correctly."
            )
        else:
            last_query = st.session_state.get("search_query", "")
            st.subheader("Results")
            for rank, (file_name, file_path, page_texts, score) in enumerate(relevant_results, start=1):
                page_number, preview = find_matching_page(page_texts, last_query)

                render_result_card(rank, file_name, score, page_number, animation_delay=(rank - 1) * 0.08)
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