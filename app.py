import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
import os
import pickle
import tempfile
import pandas as pd
import pdfplumber
import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from io import StringIO
from tqdm.auto import tqdm
from typing import List, Union
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


class Utilities:

    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API keys loaded", icon="🚀")

        return user_api_key
    
    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=["pdf"], label_visibility="collapsed", accept_multiple_files = True)
        if uploaded_file is not None:

            def show_pdf_file(uploaded_file):
                file_container = st.expander("Your PDF file :")
                for i in range(len(uploaded_file)):
                    with pdfplumber.open(uploaded_file[i]) as pdf:
                        pdf_text = ""
                        for page in pdf.pages:
                            pdf_text += page.extract_text() + "\n\n"
                    file_container.write(pdf_text)
            
            file_extension = ".pdf" 

            if file_extension== ".pdf" : 
                show_pdf_file(uploaded_file)

        else:
            st.sidebar.info(
                "👆 Upload your PDF file to get started..!"
            )
            st.session_state["reset_chat"] = True

        #print(uploaded_file)
        return uploaded_file

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature,):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder()
        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"
        with st.spinner("Processing..."):
            #uploaded_file.seek(0)
            file = uploaded_file
            
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, "Docs")

            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature,vectors)
        st.session_state["ready"] = True

        return chatbot

    def count_tokens_agent(agent, query):
        """
        Count the tokens used by the CSV Agent
        """
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens')

        return result

class Layout:
    
    def show_header(self):
        """
        Displays the header of the app
        """
        st.markdown(
            """
            <h1 style='text-align: center;'> Ask Anything: Your Personal AI Assistant</h1>
            """,
            unsafe_allow_html=True,
        )

    def show_api_key_missing(self):
        """
        Displays a message if the user has not entered an API key
        """
        st.markdown(
            """
            <div style='text-align: center;'>
                <h4>Enter your <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API key</a> to start conversation</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def prompt_form(self):
        """
        Displays the prompt form
        """
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="Ask me anything about the document...",
                key="input",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send")
            
            is_ready = submit_button and user_input
        return is_ready, user_input


class Sidebar:

    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("🧠 About")
        sections = [
            "#### Welcome to our AI Assistant, a cutting-edge solution to help you find the answers you need quickly and easily. Our AI Assistant is designed to provide you with the most relevant information from PDF sources.",
            "#### With our AI Assistant, you can ask questions on any topic, and our intelligent algorithms will search through our vast database to provide you with the most accurate and up-to-date information available. Whether you need help with a school assignment, are researching a topic for work, or simply want to learn something new, our AI Assistant is the perfect tool for you.",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature
        
    def csv_agent_button(self, uploaded_file):
        st.session_state.setdefault("show_csv_agent", False)

    def show_options(self, uploaded_file):
        with st.sidebar.expander("🛠️ Tools", expanded=False):

            self.reset_chat_button()
            self.csv_agent_button(uploaded_file)
            # self.model_selector()
            # self.temperature_slider()
            st.session_state.setdefault("model", model_name)
            st.session_state.setdefault("temperature", temperature)

original_filename="Docs"
class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, original_filename="Docs"):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        
        text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = 2000,
                chunk_overlap  = 50,
                length_function = len,
            )
        file_extension = ".pdf" #get_file_extension(original_filename)


        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)  
            data = loader.load_and_split(text_splitter)
        
            
        embeddings = OpenAIEmbeddings()

        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)


    def getDocEmbeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"
        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(file)
        #st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=500,
                             overlap=0, split_method=splitter_type)
        embeddings = OpenAIEmbeddings()
        vectors = create_retriever(embeddings, splits, retriever_type="SIMILARITY SEARCH")
        return vectors

class ChatHistory:
    
    def __init__(self):
        self.history = st.session_state.get("history", [])
        st.session_state["history"] = self.history

    def default_greeting(self):
        return "Hey! 👋"

    def default_prompt(self, topic):
        return f"Hello ! Ask me anything about {topic} 🤗"

    def initialize_user_history(self):
        st.session_state["user"] = [self.default_greeting()]

    def initialize_assistant_history(self, uploaded_file):
        st.session_state["assistant"] = [self.default_prompt(original_filename)]

    def initialize(self, uploaded_file):
        if "assistant" not in st.session_state:
            self.initialize_assistant_history(original_filename)
        if "user" not in st.session_state:
            self.initialize_user_history()

    def reset(self, uploaded_file):
        st.session_state["history"] = []
        
        self.initialize_user_history()
        self.initialize_assistant_history(original_filename)
        st.session_state["reset_chat"] = False

    def append(self, mode, message):
        st.session_state[mode].append(message)

    def generate_messages(self, container):
        con = []
        if st.session_state["assistant"]:
            with container:
                for i in range(len(st.session_state["assistant"])):
                    message(
                        st.session_state["user"][i],
                        is_user=True,
                        key=f"{i}_user",
                        avatar_style="big-smile",
                    )
                    message(st.session_state["assistant"][i], key=str(i), avatar_style="thumbs")
                    con.append("Human: " + str(st.session_state["user"][i]))
                    con.append("AI: " + str(st.session_state["assistant"][i]))
                    con.append("\n")
        return con

    def load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.history = f.read().splitlines()

    def save(self):
        with open(self.history_file, "w") as f:
            f.write("\n".join(self.history))


from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors


    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow-up entry: {question}
        Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are a friendly conversational assistant, designed to answer questions and chat with the user from a contextual file.
        You receive data from a user's files and a question, you must help the user find the information they need. 
        Your answers must be user-friendly and respond to the user.
        You will get questions and contextual information.

        question: {question}
        =========
        context: {context}
        ======="""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        retriever = self.vectors#.as_retriever()
	    
        memory = ConversationBufferMemory(memory_key="chat_history")
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT,verbose=True)
        doc_chain = load_qa_chain(llm=llm, 
                                  
                                  prompt=self.QA_PROMPT,
                                  verbose=True,
                                  chain_type= "stuff"
                                  )

        chain = ConversationalRetrievalChain(
            retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, memory=memory, verbose=True)#, return_source_documents=True)


        chain_input = {"question": query}#, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import UnstructuredPDFLoader
import PyPDF2
@st.cache_data
def load_docs(files):
    st.sidebar.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text




@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.sidebar.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=[" ", ",", "\n"])

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits
def doc_search(temperature):
        def generate(query):
            template = """Assistant is a large language model trained by OpenAI.

                    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

                    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

                    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

                    {history}
                    Human: {human_input}
                    Assistant:"""

            prompt = PromptTemplate(
                input_variables=["history", "human_input"], 
                template=template
            )


            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0), 
                prompt=prompt, 
                verbose=True, 
                memory=ConversationBufferWindowMemory(k=2),
            )

            output = chatgpt_chain.predict(human_input=query)
            return output
        def get_text():
                input_text = st.text_input("", key="input")
                return input_text 
        def prompt_form():
            """
            Displays the prompt form
            """
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Query:",
                    placeholder="Ask me anything about the document...",
                    key="input_",
                    label_visibility="collapsed",
                )
                submit_button = st.form_submit_button(label="Send")
                
                is_ready = submit_button and user_input
            return is_ready, user_input
        col1, col2 = st.columns([1,0.19])
        col1.write("Write your query here:💬")

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ['I am ready to help you!']

        if 'past' not in st.session_state:
            st.session_state['past'] = ['Hey there!']
        #user_input = get_text()
        is_ready, user_input = prompt_form()
        #is_readyy = st.button("Send")
        convo = []
        output = ""
        if is_ready: # user_input:
            output = generate(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        

        if st.session_state['generated']:

            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                convo.append("Human: " + str(st.session_state['past'][i]))
                convo.append("AI: " + str(st.session_state["generated"][i]))
                convo.append("\n")
        text_conv = '\n'.join(convo)
        # Provide download link for text file
        col2.download_button(
        label="Download Conversation",
        data=text_conv,
        file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
        mime="text/plain"
    )

       
def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖")

def main(temperature):
    # Initialize the app
    #init()

    # Instantiate the main components
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()

    layout.show_header()

    #user_api_key = utils.load_api_key()

    if not user_api_key:
        layout.show_api_key_missing()
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key

        # search = st.sidebar.button("Web Search Chat")
        # if search:
        #     doc_search()

        uploaded_file = utils.handle_upload()

        if uploaded_file:
            # Initialize chat history
            history = ChatHistory()

            # Configure the sidebar
            sidebar.show_options(uploaded_file)

            try:
                chatbot = utils.setup_chatbot(
                    uploaded_file, st.session_state["model"], st.session_state["temperature"]
                )
                st.session_state["chatbot"] = chatbot

                if st.session_state["ready"]:
                    # Create containers for chat responses and user prompts
                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        # Display the prompt form
                        is_ready, user_input = layout.prompt_form()

                        # Initialize the chat history
                        history.initialize(uploaded_file)

                        # Reset the chat history if button clicked
                        if st.session_state["reset_chat"]:
                            history.reset(uploaded_file)

                        if is_ready:
                            # Update the chat history and display the chat messages
                            history.append("user", user_input)
                            output = st.session_state["chatbot"].conversational_chat(user_input)
                            history.append("assistant", output)

                    con = history.generate_messages(response_container)
                    con = '\n'.join(con)
                    # Provide download link for text file
                    st.download_button(
                    label="Download Conversation",
                    data=con,
                    file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

    sidebar.about()



# Define a dictionary with the function names and their respective functions
functions = [
    "Select a Chat",
    "Chat with Docs",
    "Chat with AI"
]

st.set_page_config(layout="wide", page_icon="💬", page_title="AI Chatbot 🤖")
#st.markdown("# AI Chat with Docs and Web!👽")
st.markdown(
            """
            <div style='text-align: center;'>
                <h1>Chat with Docs and AI!💬</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
#st.title("")

st.subheader("Select any chat type👇")
# Create a selectbox with the function names as options
selected_function = st.selectbox("Select a Chat", functions, index = 0)
    

if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
else:
    user_api_key = st.sidebar.text_input(
            label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
        )
    if user_api_key:
        st.sidebar.success("OpenAI  API key loaded", icon="🚀")
        MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
        max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
        TEMPERATURE_MIN_VALUE = 0.0
        TEMPERATURE_MAX_VALUE = 1.0
        TEMPERATURE_DEFAULT_VALUE = 0.9
        TEMPERATURE_STEP = 0.01
        model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
        top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
        freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
        temperature = st.sidebar.slider(
                    label="Temperature",
                    min_value=TEMPERATURE_MIN_VALUE,
                    max_value=TEMPERATURE_MAX_VALUE,
                    value=TEMPERATURE_DEFAULT_VALUE,
                    step=TEMPERATURE_STEP,)

if selected_function == "Chat with Docs":
    main(temperature)
elif selected_function == "Chat with AI":
    os.environ["OPENAI_API_KEY"] = user_api_key
    doc_search(temperature)
elif selected_function == "Select a Chat":
    st.markdown(
            """
            <div style='text-align: center;'>
                <h3>Enter your OpenAI API Key First and then select a chat type!</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.warning("You haven't selected any AI Chat!!")
    
    
