import streamlit as st
import time
import os

from pypdf import PdfReader

from rag.embeddings import get_jina_embeddings
from rag.vision import describe_image
from rag.ocr import extract_text_from_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank
from rag.llm import ask_llm


st.set_page_config(
    page_title="Multimodal RAG Assistant",
    layout="wide"
)


st.markdown(
    """
    <style>
    body {
        font-family: "Inter", sans-serif;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    .panel {
        padding: 18px;
        border-radius: 14px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }

    .section-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #111827;
    }

    .stButton button {
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<div class='main-title'>Enterprise Multimodal RAG</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Retrieval-Augmented Generation over documents and images using Jina Embeddings and Groq Vision</div>",
    unsafe_allow_html=True
)


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.header("Configuration")

    groq_key_manual = st.text_input("Groq API Key", type="password")
    use_env_groq_key = st.checkbox("Use Groq key from environment (GROQ_API_KEY)", value=False)

    groq_key = os.getenv("GROQ_API_KEY", "").strip() if use_env_groq_key else groq_key_manual

    if use_env_groq_key and not groq_key:
        st.caption("GROQ_API_KEY is not set in your environment.")

    jina_key_manual = st.text_input("Jina API Key", type="password")
    use_env_jina_key = st.checkbox("Use Jina key from environment (JINA_API_KEY)", value=False)

    jina_key = os.getenv("JINA_API_KEY", "").strip() if use_env_jina_key else jina_key_manual

    if use_env_jina_key and not jina_key:
        st.caption("JINA_API_KEY is not set in your environment.")

    model = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )

    filter_type = st.radio(
        "Retrieval Scope",
        ["all", "text", "image"],
        horizontal=True
    )

    include_ocr = st.checkbox("Include OCR from uploaded image", value=True)

    st.divider()


col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Document Upload</div>", unsafe_allow_html=True)
    txt_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Image Upload</div>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)


if (txt_file or img_file) and groq_key and jina_key:

    with st.spinner("Processing knowledge sources..."):

        raw_text = ""
        if txt_file:
            if txt_file.name.endswith(".pdf"):
                reader = PdfReader(txt_file)
                raw_text = "\n".join(
                    [p.extract_text() for p in reader.pages if p.extract_text()]
                )
            else:
                raw_text = txt_file.read().decode("utf-8")

        chunks = chunk_text(raw_text) if raw_text.strip() else []
        metadata = [{"type": "text"} for _ in chunks]

        if img_file:
            image_bytes = img_file.read()
            vision_text = describe_image(image_bytes, groq_key)

            if vision_text:
                chunks.append("Image description: " + vision_text)
                metadata.append({"type": "image"})

            if include_ocr:
                ocr_text = extract_text_from_image(image_bytes)
                if ocr_text:
                    chunks.append("Image OCR text: " + ocr_text)
                    metadata.append({"type": "image"})
        if not chunks:
            retriever = None
        else:
            embeddings = get_jina_embeddings(chunks, jina_key)
            retriever = FAISSRetriever(embeddings, metadata)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Query Interface</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Enter your question",
        placeholder="Example: What does the uploaded image explain?"
    )

    run = st.button("Run Retrieval and Generate Answer", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run and query:
        if retriever is None:
            st.warning("No retrievable content found in the uploaded inputs.")
            st.stop()

        start = time.time()

        query_emb = get_jina_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type
        ids = retriever.search(query_emb, top_k=5, filter_type=f)

        retrieved_docs = [chunks[i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)

        context = "\n\n".join(reranked[:3])

        answer = ask_llm(context, query, groq_key, model)

        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Answer</div>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Performance</div>", unsafe_allow_html=True)
        st.metric("Latency (seconds)", latency)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Retrieved Context"):
            st.text(context)

        with st.expander("Recent Chat History"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"Question: {q}")
                st.markdown(f"Answer: {a}")
                st.divider()

else:
    st.info("Upload a document or image and provide API keys to begin.")
