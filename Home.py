# Importing the required modules
import os 
import streamlit as st




# Setting up Streamlit page configuration
st.set_page_config(
    page_title="AI Chatbot", layout="centered", initial_sidebar_state="expanded"
)



# Defining the main function
def main():
    # Displaying the heading of the chatbot
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 AI Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Displaying the description of the chatbot
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting with customized AI!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )



main()
