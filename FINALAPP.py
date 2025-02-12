import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, load_tools

# Check if the secrets file exists and load it
secrets_exists = os.path.exists(os.path.join(os.getcwd(), ".streamlit", "secrets.toml")) or \
                 os.path.exists(os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"))

if secrets_exists:
    load_dotenv(os.path.join(os.getcwd(), ".streamlit", "secrets.toml"))

# Function to extract text from PDFs
def extract_text_from_pdfs(docs):
    text = ""
    for doc in docs:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(doc.getbuffer())
                tmp_file_path = tmp_file.name

            extracted_text = extract_text(tmp_file_path)
            text += extracted_text

        except Exception as e:
            st.error(f"Error processing {doc.name}: {e}")
        finally:
            os.remove(tmp_file_path)

    return text

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create FAISS index
def create_faiss_index(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, 
                                          model_kwargs=model_kwargs, 
                                          encode_kwargs=encode_kwargs)

    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# Function to get the conversation chain
def get_conversation_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192",
        api_key=groq_api_key,
        streaming=True,
        verbose=True
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    retriever = MultiQueryRetriever(retriever=vector_store.as_retriever(), llm_chain=llm_chain, num_queries=3)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain, llm

# Function to get the web agent
def get_web_agent(groq_api_key):
    llm = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192",
        api_key=groq_api_key,
        streaming=True,
        verbose=True
    )
    # can create custom tools
    tools = load_tools([], llm=llm)
    from tools import summarizer_tool
    tools.append(summarizer_tool)
    
    additional_tools = load_tools(["llm-math", "google-search"], llm=llm)
    tools.extend(additional_tools)
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    ZERO_SHOT_REACT_DESCRIPTION = initialize_agent(
        agent='zero-shot-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=10,
        memory=memory,
        handle_parsing_errors=True 
    )
    return ZERO_SHOT_REACT_DESCRIPTION

# Main function
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.vector_store = None

    st.set_page_config(page_title="Multi Model Agent", page_icon=":books:")
    
    st.markdown("<h2 style='text-align: center;'>AI Agent 🤖</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('📖 API_KEYS  [REPO](https://github.com/ANeuronI/RAG-AGENT)')
        st.title("📤 Upload Pdf ")
        docs = st.file_uploader(" ", type=["pdf"], accept_multiple_files=True)
        
        file_details = []

        if docs is not None:
            for doc in docs:
                file_details.append({"FileName": doc.name})

        with st.expander("Uploaded Files"):
            if file_details:
                for details in file_details:
                    st.write(f"File Name: {details['FileName']}")

        st.subheader("Start Model🧠")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.success('Groq API key already provided!', icon='✅')
        else:
            groq_api_key = st.text_input('Enter Groq API key:', type='password', key='groq_api_key')
            if groq_api_key and (groq_api_key.startswith('gsk_') and len(groq_api_key) == 56):
                os.environ['GROQ_API_KEY'] = groq_api_key
                st.success('Groq API key provided!', icon='✅')
            else:
                st.warning('Please enter a valid Groq API key!', icon='⚠️')
        
        if st.button("Start Inference", key="start_inference") and docs:
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdfs(docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = create_faiss_index(text_chunks)
                    st.session_state.vector_store = vector_store
                    st.write("FAISS Vector Store created successfully.")
                    
                    st.session_state.conversation, llm = get_conversation_chain(vector_store, groq_api_key)
                    st.session_state.llm = llm
                    st.session_state.web_agent = get_web_agent(groq_api_key)
                else:
                    st.error("No text extracted from the documents.")
                                        
    if st.session_state.conversation:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        input_disabled = groq_api_key is None
        
        if prompt := st.chat_input("Ask your question here..." , disabled=input_disabled):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({"question": prompt})
                    if "answer" in response and "I don't know" not in response["answer"]:
                        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
                        st.write(response['answer'])
                    else:
                        with st.spinner("Searching the web..."):
                            response = st.session_state.web_agent.run(prompt)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.write(response)
                            
if __name__ == '__main__':
    main()
