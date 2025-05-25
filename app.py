import streamlit as st
from streamlit_chat import message
from io import BytesIO
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

st.set_page_config(page_title="Warehouse AI Chatbot", page_icon="📦")

st.title("📦 Warehouse AI Assistant")
st.markdown("Chat with your Inventory, Order & SOP files. Ask anything!")

uploaded_files = st.sidebar.file_uploader("Upload `inventory.csv`, `order.csv`, and `SOP` (PDF/DOCX)", accept_multiple_files=True)

@st.cache_resource
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",  # Make sure it's available in the cloud or change to HF repo
        model_type="llama",
        config={"max_new_tokens": 512, "temperature": 0.5}
    )

def load_documents(files):
    documents = []
    for file in files:
        suffix = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            path = tmp.name

        if suffix == ".csv":
            loader = CSVLoader(file_path=path, encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(file_path=path)
        elif suffix in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path=path)
        else:
            st.warning(f"❌ Unsupported file type: {file.name}")
            continue

        documents.extend(loader.load())
    return documents

if uploaded_files:
    with st.spinner("🔄 Processing your files..."):
        documents = load_documents(uploaded_files)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever()

        llm = load_llm()
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    def answer_query(query):
        result = qa_chain({"question": query, "chat_history": st.session_state.history})
        st.session_state.history.append((query, result["answer"]))
        return result["answer"]

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.past = []
        st.session_state.generated = []

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("💬 Ask your question:")
        submitted = st.form_submit_button("Send")
        if submitted and user_query:
            response = answer_query(user_query)
            st.session_state.past.append(user_query)
            st.session_state.generated.append(response)

    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i], is_user=True, key=f"user_{i}")
        message(st.session_state.generated[i], key=f"bot_{i}")
else:
    st.info("⬅️ Please upload your files to get started.")
