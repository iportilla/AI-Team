import math
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Custom RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Custom RAG Chat (Document Upload)")

# ── Sidebar: configuration ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Data Source")
    uploaded_file = st.file_uploader("Upload a README.md (or any text/markdown file)", type=["md", "txt"])
    
    st.subheader("2. System Prompt")
    system_prompt = st.text_area(
        "Edit System Prompt",
        value="You are a helpful assistant. Only use the provided context to answer the user's questions.",
        height=120
    )

    st.subheader("3. Model Configuration")
    # ── Embedding provider & model ────────────────────────────────────────────
    embedding_provider = st.selectbox(
        "Embedding Provider",
        ["Ollama (local)", "OpenAI"],
        index=0,
    )
    use_ollama_embeddings = embedding_provider == "Ollama (local)"

    if use_ollama_embeddings:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
            index=0,
        )
    else:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
        )

    st.divider()

    # ── Chat provider & model ─────────────────────────────────────────────────
    chat_provider = st.selectbox(
        "Chat Provider",
        ["Ollama (local)", "OpenAI"],
        index=0,
    )
    use_ollama_chat = chat_provider == "Ollama (local)"

    if use_ollama_chat:
        chat_model = st.selectbox(
            "Chat Model",
            ["llama3", "phi3", "mistral", "gemma"],
            index=0,
        )
    else:
        chat_model = st.selectbox(
            "Chat Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )

    st.divider()

    # ── OpenAI API Key (pre-filled from .env) ─────────────────────────────────
    env_key = os.getenv("OPEN_AI_KEY", "")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        help="Loaded from ollama/.env (OPEN_AI_KEY). Required only when using OpenAI.",
    )
    if not use_ollama_embeddings or not use_ollama_chat:
        if not openai_key:
            st.warning("⚠️ OpenAI API key required for selected provider(s).")
        elif env_key and openai_key == env_key:
            st.caption("🔑 Key loaded from `.env`")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.llm_history = [{"role": "system", "content": system_prompt}]
        st.rerun()

# ── Process Document ────────────────────────────────────────────────────────────
documents_list = []
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    raw_chunks = content.split('\n\n')
    documents_list = [chunk.strip() for chunk in raw_chunks if chunk.strip() and len(chunk.strip()) > 10]
else:
    # Try loading a default readme if present
    default_readme_path = os.path.join(os.path.dirname(__file__), "demo-readme.md")
    if os.path.exists(default_readme_path):
        with open(default_readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            raw_chunks = content.split('\n\n')
            documents_list = [chunk.strip() for chunk in raw_chunks if chunk.strip() and len(chunk.strip()) > 10]
        st.info("No file uploaded. Using `demo-readme.md` as default. You can upload your own file in the sidebar!")
    else:
        st.warning("Please upload a Markdown or text file to start.")
        st.stop()
        
if not documents_list:
    st.error("No valid text found in the document. Please upload a different file.")
    st.stop()

# ── Build clients & vector DB (cached per model selection) ────────────────────
@st.cache_resource(show_spinner="Generating embeddings for document corpus...")
def build_vector_db(use_ollama_emb: bool, emb_model: str, oai_key: str, docs: tuple):
    if use_ollama_emb:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        client = OpenAI(api_key=oai_key or "your-openai-api-key")

    vector_db = []
    for doc in docs:
        response = client.embeddings.create(input=doc, model=emb_model)
        vector_db.append({"text": doc, "embedding": response.data[0].embedding})
    return client, vector_db


@st.cache_resource
def get_chat_client(use_ollama_chat: bool, oai_key: str):
    if use_ollama_chat:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return OpenAI(api_key=oai_key or "your-openai-api-key")


try:
    # Convert list to tuple for cache hashing
    docs_tuple = tuple(documents_list)
    embedding_client, vector_db = build_vector_db(
        use_ollama_embeddings, embedding_model, openai_key, docs_tuple
    )
    dims = len(vector_db[0]["embedding"])
    st.caption(
        f"Embedding: **{embedding_model}** ({'Ollama' if use_ollama_embeddings else 'OpenAI'}) · "
        f"{dims}d vectors &nbsp;|&nbsp; "
        f"Chat: **{chat_model}** ({'Ollama' if use_ollama_chat else 'OpenAI'}) &nbsp;|&nbsp; "
        f"**{len(documents_list)} chunks** in Knowledge Base"
    )
except Exception as e:
    st.error(f"❌ Failed to initialise embedding model: {e}")
    st.info(
        "Make sure Ollama is running and the model is pulled:\n"
        f"```\nollama pull {embedding_model}\n```"
    )
    st.stop()

chat_client = get_chat_client(use_ollama_chat, openai_key)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_history" not in st.session_state:
    st.session_state.llm_history = [{"role": "system", "content": system_prompt}]
else:
    # Sync system prompt if user changes it
    if len(st.session_state.llm_history) > 0 and st.session_state.llm_history[0]["role"] == "system":
        st.session_state.llm_history[0]["content"] = system_prompt
    elif len(st.session_state.llm_history) == 0:
        st.session_state.llm_history = [{"role": "system", "content": system_prompt}]

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("context"):
            with st.expander(f"📎 Retrieved context (score: {msg.get('score', 0):.4f})"):
                st.info(msg["context"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about the document…"):
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            # 1. Embed query
            query_response = embedding_client.embeddings.create(
                input=user_input, model=embedding_model
            )
            query_embedding = query_response.data[0].embedding

            # 2. Find best matching document
            best_doc, best_score = None, -1.0
            for item in vector_db:
                score = cosine_similarity(query_embedding, item["embedding"])
                if score > best_score:
                    best_score, best_doc = score, item["text"]

            # 3. Augment prompt
            augmented = f"Context: {best_doc}\n\nUser query: {user_input}"
            current_messages = st.session_state.llm_history.copy()
            current_messages.append({"role": "user", "content": augmented})

        try:
            with st.spinner("Generating response…"):
                # 4. Generate response
                response = chat_client.chat.completions.create(
                    model=chat_model,
                    messages=current_messages,
                )
                answer = response.choices[0].message.content

            st.write(answer)
            with st.expander(f"📎 Retrieved context (score: {best_score:.4f})"):
                st.info(best_doc)

            # Update LLM history with unaugmented user message to keep it clean
            st.session_state.llm_history.append({"role": "user", "content": user_input})
            st.session_state.llm_history.append({"role": "assistant", "content": answer})

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": best_doc,
                "score": best_score,
            })

        except Exception as e:
            err = f"Error generating response: {e}"
            st.error(err)
            if use_ollama_chat:
                st.info(f"Make sure Ollama is running and `{chat_model}` is pulled:\n```\nollama pull {chat_model}\n```")
