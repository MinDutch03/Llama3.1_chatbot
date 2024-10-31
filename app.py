import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import docx2txt
import re

# Retrieve OpenAI API key from Streamlit secrets
openai_api_key = st.secrets['OPENAI_API_KEY']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []
    if 'user_questions' not in st.session_state:
        st.session_state['user_questions'] = []

def extract_number_from_query(query, default=2):
    match = re.search(r'\b(\d+)\b', query)
    return int(match.group(1)) if match else default

def conversation_chat(query, chain):
    # Check if query is asking for previous questions
    if "last" in query.lower() and "questions" in query.lower():
        # Get the number of questions requested, default to 2 if not specified
        num = extract_number_from_query(query)
        recent_questions = st.session_state['user_questions'][-num:]
        response = f"Here are your last {num} questions:\n" + "\n".join(recent_questions)
    else:
        result = chain.invoke({"question": query})
        response = result["answer"]
        # Add the question and response to history and generated responses
        st.session_state['history'].append((query, response))
        st.session_state['message_history'].append({"role": "user", "content": query})
        st.session_state['message_history'].append({"role": "assistant", "content": response})
        st.session_state['user_questions'].append(query)

    return response

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.chat_input("Ask me something....")
        if user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")

def create_conversational_chain(vector_store):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.7
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 2,
            "use_mmr": True,
            "mmr_lambda": 0.5
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        memory=memory
    )
    return chain

def main():
    initialize_session_state()
    st.set_page_config(page_title="Ask your Document")
    st.header("Ask your Document 💬")
    linkedin = "https://www.linkedin.com/in/minhduc030303/"
    st.markdown("a Multi-Documents ChatBot App by [Duc Nguyen Minh](%s) 👨🏻‍💻" % linkedin)

    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf, .docx, or .txt)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    if uploaded_files:
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

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(all_text)

        with st.spinner('Analyze Document...'):
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        chain = create_conversational_chain(vector_store)

        # Display chat history
        display_chat_history(chain)

    else:
        st.warning('⚠️ Please upload your document in the sidebar first in order to access the chatbot!')

if __name__ == "__main__":
    main()
