import streamlit as st
import boto3
import json

# --- Configuration ---
AWS_REGION = 'us-east-1'
BEDROCK_MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
KNOWLEDGE_BASE_ID = 'YBW1J8NMTI'

# --- Page Setup ---
st.set_page_config(
    page_title="Diva the Chatbot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Sidebar Layout ---
st.sidebar.title("⚙️ Settings")
with st.sidebar.expander("🧹 Tools"):
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with st.sidebar.expander("📧 Support"):
    st.markdown("[Report an issue](mailto:jeremy.gautama@derivaenergy.com)")

st.sidebar.divider()
st.sidebar.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")


# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_user_query" not in st.session_state:
    st.session_state.current_user_query = ""

# --- Bedrock Client ---
try:
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
except Exception as e:
    st.error(f"Bedrock Client Error: {e}")
    st.stop()

# --- Prompt Format ---
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

# # --- Prompt Format ---
# INSTRUCTIONAL_PREFIX = """
# Based on the following retrieved information and the user's query, please provide the most relevant details about charging guidelines.\n\nExtract and present the following in a markdown bulleted list:\n\n- **Description:**\n- **Account number:**\n- **Location:**\n- **Company ID:**\n- **Project:**\n- **Department:**\n\nIf not available, return \"N/A\".\n\nFinish with 1–2 relevant notes if needed.
# """

# --- Chat Interface ---  ⚡
# st.title("Diva The Chatbot!")
st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI Assistant for Charging Guidelines.</p>", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Input field at all times (centered first, shifts naturally after first question)
user_input = st.chat_input("Ask about codes, departments, projects, etc.")
if user_input:
    st.session_state.current_user_query = user_input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": "Thinking..."})
    st.rerun()

# --- Run Query with Bedrock ---
if st.session_state.chat_history and st.session_state.chat_history[-1]["content"] == "Thinking...":
    try:
        prev_query = st.session_state.chat_history[-2]["content"]
        full_query = INSTRUCTIONAL_PREFIX + prev_query

        response = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": full_query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{BEDROCK_MODEL_ID}"
                }
            }
        )
        answer = response["output"]["text"]
        st.session_state.chat_history[-1]["content"] = answer
        st.rerun()
    except Exception as e:
        st.session_state.chat_history[-1]["content"] = f"⚠️ Error: {e}"
        st.rerun()

# --- Footer ---
st.divider()
# st.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
