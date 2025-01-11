import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
from fpdf import FPDF

# Hardcoded Google API Key (Replace with your actual API key)
GOOGLE_API_KEY = "API_KEY"

# Streamlit App Configuration
st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("## Document Genie: Get instant insights from your Documents")

def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.

    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Concatenated text from all PDF pages.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the extracted text into smaller chunks for processing.

    Args:
        text (str): The text to be split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a vector store from the text chunks using Google's embeddings.

    Args:
        text_chunks (list): List of text chunks to be vectorized.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Creates a conversational chain for question answering.

    Returns:
        Chain: A QA chain for generating responses.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Processes the user's question and generates a response using the vector store and conversational chain.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The chatbot's response.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    # Allow dangerous deserialization since we trust the source of the data
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def save_chat_history(chat_history, file_format="txt"):
    """
    Saves the chat history to a file in the specified format.

    Args:
        chat_history (list): List of chat messages.
        file_format (str): The format of the output file (txt, pdf, or xlsx).

    Returns:
        str: Path to the saved file.
    """
    file_path = f"chat_history.{file_format}"
    if file_format == "txt":
        with open(file_path, "w", encoding="utf-8") as file:  # Use utf-8 encoding
            for message in chat_history:
                file.write(f"{message['role']}: {message['content']}\n")
    elif file_format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for message in chat_history:
            pdf.cell(200, 10, txt=f"{message['role']}: {message['content']}", ln=True)
        pdf.output(file_path)
    elif file_format == "xlsx":
        df = pd.DataFrame(chat_history)
        df.to_excel(file_path, index=False)
    return file_path

def main():
    """
    Main function to run the Streamlit app.
    """

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for the question
    user_question = st.chat_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:  # Ensure user question is provided
        # Add user's question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user's question
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get chatbot's response
        response = user_input(user_question)

        # Add chatbot's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chatbot's response
        with st.chat_message("assistant"):
            st.markdown(response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        # Save chat history
        st.title("Save Chat History")
        file_format = st.selectbox("Select file format", ["txt", "pdf", "xlsx"])
        if st.button("Save Chat History"):
            file_path = save_chat_history(st.session_state.chat_history, file_format)
            with open(file_path, "rb") as file:
                st.download_button(
                    label=f"Download Chat History as {file_format.upper()}",
                    data=file,
                    file_name=f"chat_history.{file_format}",
                    mime="text/plain" if file_format == "txt" else "application/pdf" if file_format == "pdf" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            st.success(f"Chat history saved as {file_path}!")

if __name__ == "__main__":
    main()