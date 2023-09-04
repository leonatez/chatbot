import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os
import random
import string

# Create a directory if it doesn't exist
def generate_random_string(length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string
if 'data_dir' not in st.session_state:
    st.session_state['data_dir'] = "./data"+ generate_random_string(10)
    data_dir = st.session_state['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    print('Created dir', st.session_state['data_dir'])

data_dir = st.session_state['data_dir']


# icon list here: https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
st.set_page_config(page_title="Your own chatbot", page_icon="🥷", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with your own chatbot")
industry = st.text_input('Tell me briefly about your product. For example: iPhone - A cutting edge phone with touch screen')
systemprompt = f"You are an expert on {industry} and your job is to answer all questions related to {industry}. Assume that all questions are related to the {industry} only. Keep your answers focus and based on facts, try to convince the users to buy or use the {industry} – do not hallucinate features."

#file uploader
uploaded_file = st.file_uploader('Upload a document')
# Check if a file was uploaded
if uploaded_file is not None:
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    st.write("File saved successfully.")
    

    #Initiate conversation         
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the document you just uploaded"}
        ]

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Wait a few seconds, I am learning your knowledge ⏱️"):
            reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=systemprompt))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

    index = load_data()
    #chatmode = 'condense_question' 'reAct_agent' 'OpenAI_agent'
    chatmode = 'context'
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Good question, let me think ..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
