import streamlit as st
import requests

st.set_page_config(page_title="RAG Document AI", page_icon="📄", layout="wide")
st.title("📄 RAG Document Intelligence")
st.markdown("Upload a PDF and ask questions about it!")

API_URL = "http://127.0.0.1:8005"

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Done! {data['chunks']} chunks created")
                else:
                    st.error("Upload failed!")

    st.divider()
    health = requests.get(f"{API_URL}/health").json()
    st.write("API Status:", "🟢 Ready" if health["pipeline_ready"] else "🔴 No document")

st.header("Ask a Question")
question = st.text_input("Your question:", placeholder="What are the technical skills?")

if st.button("Ask") and question:
    with st.spinner("Searching..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )
        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sources")
            for i, source in enumerate(data["sources"]):
                with st.expander(f"Source {i+1} - Page {source['page']}"):
                    st.write(source["content"])
            st.caption(f"Latency: {data['latency_ms']}ms")
        else:
            st.error(response.json().get("detail", "Error!"))
