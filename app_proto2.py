# app.py
import io
import re
from dataclasses import dataclass
from typing import List, Any, Tuple

import numpy as np
import streamlit as st
import PyPDF2
from openai import OpenAI

# --- RAG: lightweight TF‑IDF index ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    raise SystemExit("scikit-learn is required. Try: pip install scikit-learn") from e

# --- PDF export helpers (fpdf2) ---
try:
    from fpdf import FPDF
except ImportError as e:
    raise SystemExit("fpdf is required. Try: pip install fpdf") from e


# =========================
# Config & Client
# =========================
st.set_page_config(page_title="K-12 ML Support (with RAG)", page_icon="📚", layout="wide")

# Hugging Face Inference Router via OpenAI client
# Ensure you have this in .streamlit/secrets.toml:
# HUGGINGFACEHUB_ACCESS_TOKEN = "hf_..."
HUGGINGFACEHUB_ACCESS_TOKEN = st.secrets.get("HUGGINGFACEHUB_ACCESS_TOKEN", None)
if not HUGGINGFACEHUB_ACCESS_TOKEN:
    st.stop()  # Hard stop to avoid confusing downstream errors

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HUGGINGFACEHUB_ACCESS_TOKEN,
)

# Choose a router-supported chat-completions model
MODEL_NAME = "openai/gpt-oss-120b:novita"  # adjust if needed


# =========================
# PDF utilities
# =========================
def extract_text_from_pdfs(files) -> str:
    """Concatenate text from multiple PDF UploadedFile/file-like objects."""
    if not files:
        return ""
    full_text = ""
    for file in files:
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        except Exception:
            # Resilient: skip unreadable PDFs/pages
            continue
    return full_text.strip()


def extract_pages_from_single_pdf(file) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) for a single PDF file-like object."""
    pages = []
    reader = PyPDF2.PdfReader(file)
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        if t.strip():
            pages.append((i, t))
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    """Simple word-based chunker with overlap."""
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = words[i:i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks


@dataclass
class Chunk:
    id: str
    source_name: str
    page: int
    text: str


@dataclass
class RagIndex:
    vectorizer: Any
    matrix: Any
    chunks: List[Chunk]


def build_knowledge_index(knowledge_files,
                          chunk_size: int = 900,
                          overlap: int = 180) -> RagIndex:
    """Build TF‑IDF index over all knowledge PDFs (per page → sub-chunks)."""
    all_chunks: List[Chunk] = []

    for f in knowledge_files:
        # Rewind for multiple reads
        try:
            f.seek(0)
        except Exception:
            pass
        pages = extract_pages_from_single_pdf(f)
        for page_num, page_text in pages:
            for j, ch in enumerate(chunk_text(page_text, chunk_size, overlap)):
                cid = f"{getattr(f, 'name', 'file')}-p{page_num}-c{j+1}"
                src_name = getattr(f, "name", "uploaded.pdf")
                all_chunks.append(Chunk(id=cid, source_name=src_name, page=page_num, text=ch))

    if not all_chunks:
        raise ValueError("No extractable text found in Knowledge PDFs.")

    corpus = [c.text for c in all_chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(corpus)
    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks)


def retrieve(index: RagIndex, query_text: str, top_k: int = 8) -> List[Tuple[Chunk, float]]:
    if not query_text.strip():
        return []
    q_vec = index.vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, index.matrix).ravel()
    top_idx = np.argsort(-sims)[:top_k]
    results = [(index.chunks[i], float(sims[i])) for i in top_idx]
    return results


# =========================
# PDF Export Helpers
# =========================
class _PDF(FPDF):
    def header(self):
        if hasattr(self, "_title") and self._title:
            self.set_font("Helvetica", "B", 14)
            self.multi_cell(0, 8, self._title, align="C")
            self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _normalize_text(text: str) -> str:
    """Collapse excessive whitespace so it prints nicely."""
    if not text:
        return ""
    lines = [(" ".join(line.strip().split())) for line in text.splitlines()]
    s = "\n".join(lines).strip()
    return s if s else "(no content)"


def make_pdf_bytes(
    title: str,
    sections: list[tuple[str, str]],
    font_family: str = "Helvetica",
    body_size: int = 11,
    heading_size: int = 12,
) -> bytes:
    """
    sections: list of (heading, content)
    Returns raw PDF bytes.
    """
    pdf = _PDF(format="Letter", orientation="P", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf._title = title
    pdf.add_page()

    pdf.set_text_color(0, 0, 0)

    for (heading, content) in sections:
        if heading:
            pdf.set_font(font_family, "B", heading_size)
            pdf.multi_cell(0, 7, heading)
            pdf.ln(1)

        pdf.set_font(font_family, "", body_size)
        cleaned = _normalize_text(content)
        # Split paragraphs to allow some spacing
        for block in cleaned.split("\n\n"):
            pdf.multi_cell(0, 6, block)
            pdf.ln(1)

        pdf.ln(2)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def split_sections(md_text: str) -> list[tuple[str, str]]:
    """
    Simple splitter looking for 3 deliverables:
    1) Revised lesson
    2) Rationale
    3) Checklist
    Falls back to one big section if headings aren't found.
    """
    text = md_text or ""
    low = text.lower()
    idxs = []

    candidates = [
        ("Revised Lesson Plan", ["revised lesson plan", "revised lesson", "lesson plan"]),
        ("Rationale", ["rationale", "what changed and why"]),
        ("Teacher Checklist", ["checklist", "short checklist", "teacher checklist"]),
    ]

    for title, keys in candidates:
        pos = None
        for k in keys:
            p = low.find(k)
            if p != -1:
                pos = p
                break
        if pos is not None:
            idxs.append((title, pos))

    if not idxs:
        return [("Revised Lesson (Full Output)", text)]

    idxs.sort(key=lambda x: x[1])
    sections = []
    for i, (title, start) in enumerate(idxs):
        end = idxs[i + 1][1] if i + 1 < len(idxs) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections


# =========================
# Streamlit UI
# =========================
st.title("📚 K-12 ML Support (with RAG)")
st.markdown("Upload **Knowledge PDFs** (indexed) and **Lesson PDFs** (to revise/enhance).")

with st.sidebar:
    st.subheader("RAG Settings")
    k_top = st.slider("Top‑K chunks", 3, 20, 8, step=1)
    chunk_size = st.slider("Chunk size (words)", 400, 1500, 900, step=50)
    overlap = st.slider("Chunk overlap (words)", 50, 400, 180, step=10)
    show_snippets = st.checkbox("Show retrieved snippets", value=True)

knowledge_files = st.file_uploader(
    "📘 Upload Knowledge PDFs (for context)",
    type="pdf",
    accept_multiple_files=True
)

lesson_files = st.file_uploader(
    "📗 Upload Lesson PDFs (to revise or enhance)",
    type="pdf",
    accept_multiple_files=True
)

companion_files = st.file_uploader(
    "📙 Upload Lesson Companion Materials (optional, PDF)",
    type="pdf",
    accept_multiple_files=True
)

custom_instruction = st.text_area(
    "✏️ Optional: Add custom instructions for revising the lesson",
    placeholder="E.g., 'Make the lesson easier for high school students' or 'Align with AI ethics principles'"
)


# Cache the RAG index so we don’t rebuild on every click
@st.cache_resource(show_spinner=False)
def _cached_index(_files_bytes_and_names, chunk_size, overlap):
    # We need to rebuild file-like objects because cached data is bytes
    rebuild = []
    for name, file_bytes in _files_bytes_and_names:
        f = io.BytesIO(file_bytes)
        f.name = name
        rebuild.append(f)
    return build_knowledge_index(rebuild, chunk_size=chunk_size, overlap=overlap)


# =========================
# Action
# =========================
if st.button("🔁 Revise Lessons"):
    if not knowledge_files:
        st.warning("Please upload at least one Knowledge PDF.")
    elif not lesson_files:
        st.warning("Please upload at least one Lesson PDF.")
    else:
        with st.spinner("Indexing knowledge, retrieving evidence, and revising lessons..."):
            try:
                # --- Build or reuse index (cache-friendly) ---
                k_files_serialized = [(f.name, f.read()) for f in knowledge_files]
                index = _cached_index(k_files_serialized, chunk_size, overlap)

                # --- Extract lessons/companion text ---
                lesson_text = extract_text_from_pdfs(lesson_files)
                companion_text = extract_text_from_pdfs(companion_files) if companion_files else ""

                # --- Validate ---
                if not lesson_text.strip() and not companion_text.strip():
                    st.warning("No extractable text found in Lesson PDFs or companion materials.")
                    st.stop()

                # --- Combine lesson + companion for retrieval/prompt ---
                combined_lesson_text = lesson_text
                if companion_text.strip():
                    combined_lesson_text = (
                        f"{lesson_text}\n\n--- COMPANION MATERIALS ---\n{companion_text}"
                    )

                # --- Retrieve top‑K chunks using the combined text ---
                hits = retrieve(index, combined_lesson_text, top_k=k_top)

                # --- Prepare numbered knowledge blocks with inline tags ---
                numbered_blocks = []
                for i, (chunk, score) in enumerate(hits, start=1):
                    tag = f"[K{i}]"
                    header = f"{tag} {chunk.source_name} • p.{chunk.page}"
                    snippet = chunk.text.strip()
                    # Normalize whitespace
                    snippet = re.sub(r"\s+\n", "\n", snippet)
                    snippet = re.sub(r"\n\s+", "\n", snippet)
                    snippet = re.sub(r"\s{2,}", " ", snippet)
                    numbered_blocks.append((tag, header, snippet, score))

                if show_snippets and numbered_blocks:
                    st.subheader("Retrieved knowledge")
                    for tag, header, snippet, score in numbered_blocks:
                        with st.expander(f"{header}  (similarity={score:.3f})"):
                            st.write(snippet)

                # --- Build final prompt with citations guidance ---
                knowledge_block_text = "\n\n".join(
                    f"{tag} {header}\n{snippet}"
                    for tag, header, snippet, _ in numbered_blocks
                )

                pedagogy_block = """
You are revising lesson materials for real classrooms.

USE THE KNOWLEDGE SOURCES FIRST. When you assert a fact that comes from a source, add the inline tag (e.g., [K1]) right after the sentence.
If multiple sources support a statement, include multiple tags (e.g., [K1][K3]).

Deliverables:
1) A revised lesson plan (concise and implementable today).
2) A brief rationale (what changed and why) tied to sources via [Ki] tags.
3) A short checklist for the teacher.

Design principles:
- Active, student-centered, inquiry-based, differentiated.
- Offer either Gradual Release or 5E flow (teacher chooses).
- Always include:
  • Summary/description
  • Grade level(s), subject(s), duration
  • Objectives & standards
  • Materials & student tech (if any)
  • Engaging hook
  • Procedure
  • Assessment
  • Differentiation & extensions
  • Glossary of key terms
  • Citations (use [Ki] tags inline)

If sources are insufficient, say so and propose safe, clearly-labeled general best practices (without fabricating citations).
"""

                final_user_prompt = (
                    f"Knowledge sources (ranked):\n\n{knowledge_block_text}\n\n"
                    f"Lesson(s) to revise and companion materials:\n\n{combined_lesson_text}\n\n"
                )

                if custom_instruction.strip():
                    final_user_prompt += f"Additional teacher instruction: {custom_instruction.strip()}\n\n"

                final_user_prompt += "Now produce the revised lesson, rationale, and checklist with inline [Ki] citations."

                # --- Call HF Router via OpenAI client ---
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": pedagogy_block},
                        {"role": "user", "content": final_user_prompt},
                    ],
                )
                answer = response.choices[0].message.content

                # --- Display result ---
                st.success("📝 Revised Lesson (RAG‑enhanced)")
                st.markdown(answer)

                # --- Build and offer PDFs ---
                sections = split_sections(answer)

                # Combined PDF
                pdf_bytes = make_pdf_bytes(
                    title="K-12 ML Support (RAG‑enhanced)",
                    sections=sections
                )
                st.download_button(
                    label="⬇️ Download PDF (Combined)",
                    data=pdf_bytes,
                    file_name="revised_lesson_RAG.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                # Individual PDFs
                if sections:
                    st.markdown("**Per‑section PDFs:**")
                    cols = st.columns(min(3, len(sections)))
                    for i, (heading, content) in enumerate(sections):
                        b = make_pdf_bytes(
                            title=f"K-12 ML Support • {heading}",
                            sections=[(heading, content)]
                        )
                        with cols[i % len(cols)]:
                            st.download_button(
                                label=f"⬇️ {heading} (PDF)",
                                data=b,
                                file_name=f"{heading.lower().replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )

            except Exception as e:
                st.error(f"An error occurred: {e}")
