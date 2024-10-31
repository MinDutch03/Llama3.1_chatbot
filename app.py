import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import docx2txt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import tempfile
from huggingface_hub import login

groq_api_key = st.secrets['GROQ_API_KEY']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []  # Store all conversation history

def conversation_chat(query, chain, history):
    # Store user input in message_history
    st.session_state['message_history'].append({"role": "user", "content": query})
    # Get response from the conversational chain
    result = chain.invoke({"question": query, "chat_history": history})
    # Append both user input and bot response to history
    history.append((query, result["answer"]))
    # Store bot response in message_history
    st.session_state['message_history'].append({"role": "assistant", "content": result["answer"]})
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.chat_input("Ask me something....")

        if user_input:
            with st.spinner('Generating response...'):
                # Call conversation_chat to handle conversation logic and response
                output = conversation_chat(user_input, chain, st.session_state['history'])
                # Append user input and bot output to past and generated lists
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    # Display entire chat history
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")

def create_conversational_chain(vector_store):
    # Create llm
    llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='mixtral-8x7b-32768'
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Set up the chain with MMR-based retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 2,
            "use_mmr": True,          # Enable Maximal Marginal Relevance
            "mmr_lambda": 0.5         # Adjust lambda to balance relevance (0) vs. diversity (1)
        }
    )
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=retriever, memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.set_page_config(page_title="Ask your Document")
    st.header("Ask your Document 💬")
    linkedin = "https://www.linkedin.com/in/minhduc030303/"
    st.markdown("a Multi-Documents ChatBot App by [Duc Nguyen Minh](%s) 👨🏻‍💻" % linkedin)
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf, .docx, or .txt)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

    if uploaded_files:
        # extract text from uploaded files
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(uploaded_file)
            else:
                st.write(f"Unsupported file type: {uploaded_file.name}")
                continue
            all_text += text

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=500, length_function=len)
        text_chunks = text_splitter.split_text(all_text)

        with st.spinner('Analyze Document...'):
            # Create vector store
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)
    else:
        st.warning('⚠️ Please upload your document in the sidebar first in order to access the chatbot!')

if __name__ == "__main__":
    main()
