import streamlit as st
import boto3
import json

# --- 1. Configuration (Hardcoded as per request) ---
# IMPORTANT: For production, consider using AWS Secrets Manager or environment variables
# instead of hardcoding sensitive IDs directly in your script.
AWS_REGION = 'us-east-1' # Your AWS Region (e.g., 'us-east-1', 'us-west-2')
# The Model ID for Claude 3.5 Sonnet (recommended for RAG)
BEDROCK_MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
# Your AWS Bedrock Knowledge Base ID
KNOWLEDGE_BASE_ID = 'YBW1J8NMTI' # <-- REPLACE THIS with your actual Knowledge Base ID

# --- Page Configuration ---
st.set_page_config(
    page_title="Charging Guidelines Assistant",
    layout="centered", # 'centered' layout for a focused experience
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a clean, modern look (Shadcn-like principles) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    /* Overall page background */
    .stApp {
        background-color: #f8fafc; /* Tailwind gray-50 */
    }

    /* Main title styling */
    h1 {
        color: #1e3a8a; /* A deep blue */
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        font-size: 2.5rem; /* Larger for impact */
    }

    /* Welcome subheader */
    .welcome-subheader {
        text-align: center;
        color: #555;
        font-size: 1.15rem;
        margin-bottom: 2.5rem; /* Adjusted margin as buttons are removed */
        line-height: 1.6;
    }

    /* Suggested questions container (now unused, but keeping CSS for robustness) */
    .suggested-questions-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem; /* Space between buttons */
        justify-content: center;
        margin-bottom: 0rem; /* No margin needed if questions are gone */
    }

    /* Suggested question button styling (now unused, but keeping CSS for robustness) */
    .stButton > button {
        background-color: #ffffff;
        color: #3b82f6; /* Tailwind blue-500 */
        border: 1px solid #d1d5db; /* Tailwind gray-300 */
        border-radius: 0.5rem; /* Rounded corners */
        padding: 0.75rem 1.25rem; /* Comfortable padding */
        font-weight: 500;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* Subtle shadow */
    }
    .stButton > button:hover {
        background-color: #eff6ff; /* Tailwind blue-50 */
        border-color: #2563eb; /* Tailwind blue-600 */
        color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* More pronounced shadow on hover */
    }

    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid #d1d5db !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05) !important;
        font-size: 1rem !important;
    }
    .stTextInput > label {
        display: none; /* Hide default label for chat input */
    }

    /* Chat history container (to enable scrolling) */
    .chat-history-container {
        height: calc(100vh - 200px); /* Adjust height based on header/footer/input */
        overflow-y: auto;
        padding-right: 1rem; /* Space for scrollbar */
        padding-bottom: 1rem; /* Space from bottom */
        display: flex;
        flex-direction: column;
        gap: 0.75rem; /* Space between messages */
    }

    /* Specific styling for st.chat_message elements */
    .st-chat-message-user, .st-chat-message-assistant {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem; /* More rounded corners for bubbles */
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); /* Subtle shadow for bubbles */
        max-width: 75%; /* Limit bubble width */
    }
    .st-chat-message-user {
        background-color: #dbeafe; /* Tailwind blue-100 */
        align-self: flex-end;
        margin-left: auto; /* Push to right */
    }
    .st-chat-message-assistant {
        background-color: #ffffff; /* White for assistant */
        align-self: flex-start;
        margin-right: auto; /* Push to left */
    }
    .st-chat-message-user > div[data-testid="stChatMessageContent"] {
        color: #1e40af; /* Darker blue text for user */
    }
    .st-chat-message-assistant > div[data-testid="stChatMessageContent"] {
        color: #333; /* Dark gray text for assistant */
    }

    /* Fixed position for chat input at bottom */
    div.st-emotion-cache-1kyxreq { /* This targets the st.chat_input container */
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8fafc; /* Match body background */
        padding: 1rem 0;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.05); /* Shadow above input */
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div.st-emotion-cache-1kyxreq > div {
        width: 90%; /* Control width of the input field */
        max-width: 800px; /* Max width for larger screens */
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Chat History and State Variables ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_chat_interface" not in st.session_state:
    st.session_state.show_chat_interface = False
if "current_user_query" not in st.session_state:
    st.session_state.current_user_query = ""

# --- Initialize Bedrock Client ---
bedrock_agent_runtime = None
try:
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
except Exception as e:
    st.error(f"Error initializing AWS Bedrock client: {e}")
    st.warning("Please ensure your AWS credentials are configured and the region is correct.")
    st.stop() # Stop if client cannot be initialized

# --- Define the instructional prefix text ---
# This text will be prepended to the user's query before sending to Bedrock
# Explicitly instruct for newlines after each bullet point.
INSTRUCTIONAL_PREFIX = """
Based on the following retrieved information and the user's query, please provide the most relevant details about charging guidelines.
Specifically, extract and present the following information in a bulleted list format, with each item on a new line, enter, having space between each bullet point:

-   **Description:** A concise summary of the charging activity or guideline.\n
-   **Account number:** The associated account number.\n
-   **Location:** The relevant location for this guideline/activity.\n
-   **Company ID:** The company identification number.\n
-   **Project:** The project name or code.\n
-   **Department:** The department responsible or associated with this.\n

If any of these specific pieces of information are not found in the retrieved context, please state "N/A" or "Not applicable" for that item.
Do not include any other conversational text or preamble; just the bulleted list.

In addition, please ensure that the response is formatted in a way that is easy to read and understand, with clear separation between each bullet point.

In addition, add a sentence or two at the end of the response for any relevant information that may not fit into the bullet points, but is still important for the user to know.

""" # Added extra newline for better separation, though not strictly necessary if you rely on the model.

# --- Main Interface Logic ---
if not st.session_state.show_chat_interface:
    # --- Welcome Screen ---
    st.markdown("<h1 class='main-header'>💡 Charging Guidelines Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='welcome-subheader'>How can I help you today with regards to Charging Guidelines?</p>", unsafe_allow_html=True)

    # Main text input for welcome screen (centered)
    main_input_col = st.columns([1, 4, 1])
    with main_input_col[1]:
        user_input_welcome = st.text_input(
            "Ask a question...",
            key="welcome_input",
            placeholder="e.g., What is the activity code for home charging?",
            label_visibility="collapsed"
        )
        if user_input_welcome:
            st.session_state.current_user_query = user_input_welcome
            st.session_state.show_chat_interface = True
            st.rerun()

else:
    # --- Chat Interface ---
    st.markdown("<h1 class='main-header'>💡 Charging Guidelines Assistant</h1>", unsafe_allow_html=True)

    # Chat display area using st.container for a card-like effect
    with st.container(height=600, border=True): # Fixed height container for chat history
        # Use st.chat_message for cleaner chat bubbles
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

    # --- Chat Input (at the bottom, handled by st.chat_input's fixed position) ---
    user_input_chat = st.chat_input(
        "Ask about charging guidelines, activity codes, or specific scenarios...",
        key="chat_input_main"
    )
    if user_input_chat:
        st.session_state.current_user_query = user_input_chat
        st.rerun()

# --- Handle Query Processing (if a query is active) ---
if st.session_state.current_user_query:
    user_query = st.session_state.current_user_query

    # Clear current_user_query to prevent re-processing on rerun
    st.session_state.current_user_query = ""

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    # Add a "Thinking..." message placeholder
    st.session_state.chat_history.append({"role": "assistant", "content": "Thinking..."})
    st.rerun() # Rerun to display the new messages immediately

# If the last message is "Thinking...", try to get a response from Bedrock
if st.session_state.chat_history and st.session_state.chat_history[-1]["content"] == "Thinking...":
    if KNOWLEDGE_BASE_ID == "YOUR_KNOWLEDGE_BASE_ID":
        st.session_state.chat_history[-1]["content"] = "⚠️ Please replace 'YOUR_KNOWLEDGE_BASE_ID' in the code with your actual Knowledge Base ID."
        st.rerun()
    else:
        try:
            # Get the actual user query (it's always the one before "Thinking...")
            user_query_to_process = st.session_state.chat_history[-2]["content"]

            # PREPEND THE INSTRUCTIONAL_PREFIX TO THE USER'S QUERY
            final_query_for_bedrock = INSTRUCTIONAL_PREFIX + user_query_to_process

            response = bedrock_agent_runtime.retrieve_and_generate(
                input={"text": final_query_for_bedrock}, # Send the combined query
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                        "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{BEDROCK_MODEL_ID}"
                        # Removed "generationConfiguration" as it's no longer needed for this approach
                    }
                }
            )

            answer = response["output"]["text"]
            st.session_state.chat_history[-1]["content"] = answer
            st.rerun()

        except Exception as e:
            st.session_state.chat_history[-1]["content"] = f"⚠️ Error: {e}"
            st.rerun()

# --- Clear Chat Button (visible only in chat interface) ---
if st.session_state.show_chat_interface:
    # Place clear chat button at the top right of the chat container for easy access
    with st.container(): # Use a container to place the button
        col1, col2 = st.columns([0.8, 0.2])
        with col2:
            if st.button("Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.session_state.show_chat_interface = False # Go back to welcome screen
                st.rerun()
