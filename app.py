import streamlit as st
from langchain_integration import LangChainRAGSystem

# Initialize your RAG system
rag_system = LangChainRAGSystem()

# Inject custom CSS for UChicago theme and adjust layout
st.markdown(
    """
    <style>
    /* Override Streamlit dark theme background if needed */
    .main {
        background-color: #FFFFFF; /* Set a white background */
        color: #333333;
    }

    /* Title styling */
    h1, h2, h3 {
        color: #800000; /* UChicago Maroon */
        font-family: 'Georgia', serif;
        font-weight: bold;
    }

    /* Chat history title */
    h3 {
        color: #800000;
        font-size: 18px;
        margin-top: 20px;
    }

    /* Input box styling */
    .stTextInput > div > input {
        border: 2px solid #800000; /* Maroon border */
        background-color: #FFFFFF; /* White background for input */
        color: #333333; /* Dark gray text */
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #800000; /* Maroon background */
        color: #FFFFFF; /* White text */
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 16px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #600000; /* Darker maroon on hover */
    }

    /* Message bubbles for chat (for user and bot messages) */
    .user-bubble {
        background-color: #F0F0F0; /* Light gray for user */
        color: #333333;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }

    .bot-bubble {
        background-color: #800000; /* Maroon for bot */
        color: #FFFFFF;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("University of Chicago Data Science Chatbot")
st.write("Ask about admission requirements, capstone projects, advising, or transcript submissions!")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Callback function to clear the input after submission
def submit_message():
    # Retrieve user input
    user_input = st.session_state.user_input
    
    if user_input:
        # Get response from the RAG system
        response = rag_system.query(user_input, use_memory=True)
        if not response:
            print("No response received from RAG system.")
        print("*"*50)
        print("## Response:", response)
        print("*"*50)
        # Extract answer and best source document
        answer = response.get('answer', "No answer found.")
        source_document = response.get('source_document', None)
        
        # Append user input and bot response to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))
        
        # Append only the URL from the best source document to chat history for context
        if source_document:
            doc_content = source_document["content"]
            doc_url = source_document.get("metadata", {}).get("url", "URL not available")
            st.session_state.chat_history.append((
                "Source",
                f"URL: {doc_url}"
            ))

        # Clear the input field
        st.session_state.user_input = ""  # Reset the input field using session state

# Display chat history in a scrollable container
st.markdown("### Chat History")
with st.container():
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            with st.chat_message("user"):
                st.write(f"**{speaker}:** {message}")
        else:
            with st.chat_message("assistant"):
                st.write(f"**{speaker}:** {message}")

# Message input box at the bottom, linked with submit_message callback
st.text_input("Your Message", key="user_input", on_change=submit_message)
st.button("Send", on_click=submit_message)



