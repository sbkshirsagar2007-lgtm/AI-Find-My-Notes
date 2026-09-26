# 📚 AI Find My Notes

Intelligent academic notes retrieval using **TF-IDF** and **Cosine Similarity**.

Type a topic or question (e.g. *"Explain heuristic search and shortest path"*) and the app
finds which of your PDF notes covers it — and the **exact page number** it's on. No
internet connection, no API keys, and no downloaded AI model required; the matching is
done entirely with classic, well-understood information-retrieval maths.

---

## ✨ Features

- 🔍 Natural-language topic search across all your PDF notes
- 📄 Pinpoints the **exact page** where a topic is actually discussed
- ⭐ Similarity score shown for every result, so you can judge match strength
- 📖 Built-in PDF viewer that jumps straight to the matched page
- ⬇️ One-click download for any matched PDF
- 🩺 "PDF scan details" panel to debug why a file isn't being found (e.g. scanned/image-only PDFs)
- 🎨 Custom dark, olive-green themed UI with an offline (fully embedded) background image
- 🔌 Works completely offline — no internet, no API key, no paid service

## 🧠 How It Works (short version)

1. **Scan** — every PDF inside `dataset/` is discovered automatically.
2. **Extract** — each PDF's text is read page-by-page using `pypdf`.
3. **Clean** — text is lowercased and normalised so punctuation/casing don't affect matching.
4. **Index** — all documents are cached (`st.cache_data`) so this only happens once.
5. **Rank** — your query and every document are converted into TF-IDF vectors, and ranked
   by cosine similarity.
6. **Pinpoint** — for each top match, every page is scored by keyword occurrence count, and
   the highest-scoring page is reported as the answer.

Full technical breakdown, algorithm explanations, and viva Q&A are in
`AI_Find_My_Notes_Project_Documentation.pdf`.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| PDF reading | [pypdf](https://pypdf.readthedocs.io) |
| Search / ranking | [scikit-learn](https://scikit-learn.org) (`TfidfVectorizer`, `cosine_similarity`) |
| PDF viewer | [streamlit-pdf-viewer](https://pypi.org/project/streamlit-pdf-viewer/) |
| Language | Python 3 |

## 📁 Folder Structure

```
ai-find-my-notes/
├── app.py                # main application
├── requirements.txt      # Python dependencies
├── README.md
├── dataset/               # put your PDF notes here (scanned automatically)
│   ├── subject1_notes.pdf
│   └── subject2_notes.pdf
└── assets/                # background images used by the UI
    ├── bg-books-teal.png
    ├── bg-cozy-flatlay.png
    ├── bg-bookshelf-wide.png
    └── bg-open-book.png
```

## 🚀 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ai-find-my-notes.git
   cd ai-find-my-notes
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your notes**
   Drop your PDF files into the `dataset/` folder (subfolders are scanned too).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## 🖱️ Usage

1. Type a topic or question in the search box.
2. Click **Search**.
3. Review the ranked results — each shows a similarity score and the page number the
   topic was found on.
4. Click **View PDF** to open the embedded viewer, scrolled straight to that page, or
   **Download PDF** to save the file.
5. If a PDF isn't showing up in results, open **PDF scan details** to check whether it's
   a scanned/image-only PDF with no extractable text.

## ⚠️ Limitations

- Matches on actual word/phrase overlap — it doesn't understand synonyms or paraphrased
  meaning (no semantic/AI model is used).
- Scanned/image-only PDFs with no real text layer can't be searched unless OCR is added.
- Designed for a single user's local notes, not a multi-user/cloud system.

## 🔭 Future Improvements

- OCR support (`pytesseract` + `pdf2image`) for scanned PDFs
- Optional semantic search layer (sentence embeddings) for synonym/paraphrase matching
- Support for `.docx` and `.pptx` notes, not just PDFs

## 🌐 Live Demo

Deployed on Streamlit Community Cloud: `<add your app link here after deploying>`

## 👤 Author

**Santa** — Diploma in Computer Engineering (Final Year), Government Polytechnic, Mumbai