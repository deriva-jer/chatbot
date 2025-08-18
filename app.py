import os
import json
import uuid
import boto3
import streamlit as st
from typing import List, Dict

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "YBW1J8NMTI")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- LangChain (custom implementation for existing table) ---
from langchain_aws.chat_models import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory  # Correct import path
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
from typing import List

# Custom DynamoDB Chat History for your existing table schema
class CustomDynamoDBChatHistory(BaseChatMessageHistory):
    def __init__(self, table_name: str, session_id: str):
        super().__init__()
        self.table_name = table_name
        self.session_id = session_id
        self.dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        self.table = self.dynamodb.Table(table_name)
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from DynamoDB"""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id),
                ScanIndexForward=True  # Sort by timestamp ascending
            )
            
            messages = []
            for item in response.get('Items', []):
                msg_type = item.get('message_type', 'human')
                content = item.get('content', '')
                
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            return messages
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
            return []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to DynamoDB"""
        try:
            msg_type = 'human' if isinstance(message, HumanMessage) else 'ai'
            self.table.put_item(
                Item={
                    'session_id': self.session_id,
                    'message_timestamp': str(int(time.time() * 1000)),  # Convert to string
                    'message_type': msg_type,
                    'content': message.content
                }
            )
        except Exception as e:
            st.error(f"Failed to save message: {e}")
    
    def add_user_message(self, message: str):
        """Add user message to DynamoDB"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        """Add AI message to DynamoDB"""
        self.add_message(AIMessage(content=message))
    
    def clear(self):
        """Clear all messages for this session"""
        try:
            # Query all items for this session
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id)
            )
            
            # Delete each item
            for item in response.get('Items', []):
                self.table.delete_item(
                    Key={
                        'session_id': item['session_id'],
                        'message_timestamp': str(item['message_timestamp'])  # Ensure string format
                    }
                )
        except Exception as e:
            st.error(f"Failed to clear history: {e}")

# --- AWS Clients ---
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# --- Create DynamoDB table if it doesn't exist ---
def create_dynamodb_table_if_not_exists():
    try:
        table = dynamodb.Table(DDB_TABLE_NAME)
        table.load()  # Check if table exists
        # st.success(f"✅ DynamoDB table '{DDB_TABLE_NAME}' exists")
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        st.warning(f"Creating DynamoDB table '{DDB_TABLE_NAME}'...")
        try:
            table = dynamodb.create_table(
                TableName=DDB_TABLE_NAME,
                KeySchema=[
                    {
                        'AttributeName': 'SessionId',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'SessionId',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.wait_until_exists()
            st.success(f"✅ Created DynamoDB table '{DDB_TABLE_NAME}'")
        except Exception as e:
            st.error(f"❌ Failed to create table: {e}")
            return False
    except Exception as e:
        st.error(f"❌ Error checking table: {e}")
        return False
    return True

# Check/create table on startup
if not create_dynamodb_table_if_not_exists():
    st.stop()

# --- Streamlit Page ---
st.set_page_config(page_title="Diva the Chatbot", layout="centered", initial_sidebar_state="expanded")

st.sidebar.title("⚙️ Settings")

# Session / User identity (swap with SSO user if you have it)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def reset_history():
    try:
        # Use custom history class
        hist = CustomDynamoDBChatHistory(
            table_name=DDB_TABLE_NAME, 
            session_id=st.session_state["session_id"]
        )
        hist.clear()
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

with st.sidebar.expander("🧹 Tools", expanded=True):
    if st.button("🗑️ Clear Chat"):
        reset_history()
        st.rerun()

with st.sidebar.expander("📧 Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

st.sidebar.divider()
st.sidebar.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")

st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot for Charging Guidelines.</p>", unsafe_allow_html=True)

# --- LLM + Memory (simplified approach) ---
chat_model = ChatBedrock(
    client=bedrock_runtime,
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
    # model_kwargs={"temperature": 0.2}
)

# Use custom DynamoDB chat history that works with your table schema
chat_history_store = CustomDynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_history_store,
    return_messages=True
)

# --- Retrieval from Bedrock KB ---
def retrieve_from_kb(query: str, max_results: int = 6) -> Dict:
    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": max_results}}
    )
    chunks, sources = [], []
    for item in resp.get("retrievalResults", []):
        txt = item.get("content", {}).get("text", "")
        loc = item.get("location", {})
        score = item.get("score", None)
        if txt:
            chunks.append(txt.strip())
        if loc:
            sources.append({"location": loc, "score": score})
    return {"context": "\n\n---\n\n".join(chunks), "sources": sources}

# # --- Clarifying Router ---
# ROUTER_POLICY = """
# You are Diva, Deriva's internal charging-guidelines assistant.

# Goal: Decide if you should ASK CLARIFYING QUESTIONS first OR ANSWER now.

# Critical fields that change the answer:
# - team (e.g., Operations, Engineering, Finance, IT)
# - If team == Operations: asset_type (Wind, Solar, Battery)
# - site/plant name (if the guideline varies by site)
# - optionally: account number, project, department if user explicitly needs the 6-field list

# IMPORTANT: For vague queries like "travel", "expenses", "where should I charge", "what code to use" - ALWAYS ask clarifying questions first.

# Rules:
# 1) If it's just a greeting (hi, hello, hey) → answer directly with greeting
# 2) If query is vague or general (travel, expenses, charging codes) → ask clarifying questions
# 3) If team is unknown AND the answer depends on team → ask a short clarifying question
# 4) If team=Operations but asset_type unknown → ask which (Wind/Solar/Battery)
# 5) If site materially affects the answer and it's missing → ask which site
# 6) Ask at most TWO concise questions in one turn
# 7) If information is sufficient → do NOT ask questions; proceed to answer

# Return ONLY a JSON object with:
# {
#   "intent": "clarify" | "answer",
#   "questions": [ "q1", "q2" ],
#   "known": {"team": "...", "asset_type": "...", "site": "..."},
#   "notes": ""
# }
# """

ROUTER_POLICY = """
You are Diva, Deriva's internal charging-guidelines assistant.

DEFAULT BEHAVIOR: Clarify first for any query about policies, charging, codes, expenses, departments, projects, or sites. 
Only skip clarification when:
- It's a simple greeting (hi/hello/hey), OR
- The user message (plus chat history) already provides all critical fields needed to answer unambiguously.

Critical fields:
- team (Operations, Engineering, Finance, IT)
- if team == Operations: asset_type (Wind, Solar, Battery)
- site/plant name if the policy can vary by site (assume it might vary unless clearly org-wide)
- department/project if the user explicitly wants the 6-field list and those are missing

Rules:
1) If it's just a greeting → intent: "answer".
2) Otherwise, prefer intent: "clarify" and ask at most TWO concise, targeted questions for the missing critical fields.
3) If prior chat history already contains the needed fields, intent: "answer".
4) Never invent values; if unsure, ask.
5) Keep questions short and friendly.

Return ONLY a JSON object:
{
  "intent": "clarify" | "answer",
  "questions": ["q1", "q2"],
  "known": {"team": "...", "asset_type": "...", "site": "...", "department": "...", "project": "..."},
  "notes": ""
}
"""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_POLICY),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# def route_turn(user_input: str) -> Dict:
#     try:
#         mv = memory.load_memory_variables({})
#         # Use a simpler approach - just pass the messages directly without cleaning
#         msgs = router_prompt.format_messages(chat_history=mv.get("chat_history", []), input=user_input)
#         resp = chat_model.invoke(msgs)
        
#         # Clean up the response content and try to parse JSON
#         content = resp.content.strip()
        
#         # Remove any markdown code block formatting
#         if content.startswith("```json"):
#             content = content[7:]
#         if content.startswith("```"):
#             content = content[3:]
#         if content.endswith("```"):
#             content = content[:-3]
        
#         # Remove any leading/trailing whitespace and newlines
#         content = content.strip()
        
#         # Try to find JSON in the content
#         import re
#         json_match = re.search(r'\{.*\}', content, re.DOTALL)
#         if json_match:
#             content = json_match.group()
        
#         data = json.loads(content)
#         if "intent" in data and data["intent"] in ("clarify", "answer"):
#             return data
#     except Exception as e:
#         # Remove the warning message and just silently default to answer
#         pass
    
#     # Default fallback
#     return {"intent": "answer", "questions": [], "known": {}, "notes": ""}


def route_turn(user_input: str) -> Dict:
    # quick greeting check (let greetings bypass clarification)
    text = (user_input or "").strip().lower()
    GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text in GREETINGS or any(text.startswith(g) for g in GREETINGS):
        return {"intent": "answer", "questions": [], "known": {"reason": "greeting"}, "notes": ""}

    try:
        mv = memory.load_memory_variables({})
        msgs = router_prompt.format_messages(chat_history=mv.get("chat_history", []), input=user_input)
        resp = chat_model.invoke(msgs)
        content = resp.content.strip()

        # strip code fences if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        import re, json as _json
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            data = _json.loads(m.group())
        else:
            data = _json.loads(content)

        # Safety guard: enforce clarify-first unless we have enough info
        intent = data.get("intent", "clarify")
        known = data.get("known", {}) or {}

        # Determine sufficiency of known fields
        team = (known.get("team") or "").strip().lower()
        asset_type = (known.get("asset_type") or "").strip().lower()
        site = (known.get("site") or "").strip()

        # Minimal sufficiency rule:
        # - If team is not provided → clarify
        # - If team is Operations and asset_type missing → clarify
        # - Otherwise allow answer
        sufficient = False
        if team:
            if team == "operations":
                sufficient = bool(asset_type)
            else:
                sufficient = True  # non-Operations answers usually don't need asset_type

        if intent == "answer" and not sufficient:
            # Flip to clarify with targeted questions
            questions = []
            if not team:
                questions.append("which team you're with (Operations, Engineering, Finance, or IT)")
            if team == "operations" and not asset_type:
                questions.append("if it's for Wind, Solar, or Battery (and the site/plant if applicable)")
            if not questions:
                questions = ["which team you're with (Operations, Engineering, Finance, or IT)",
                             "if it's for Wind, Solar, or Battery (and the site/plant if applicable)"]
            return {"intent": "clarify", "questions": questions[:2], "known": known, "notes": "insufficient context"}

        # Default to clarify-first if model didn’t explicitly decide
        if intent not in ("clarify", "answer"):
            intent = "clarify"

        # Strong default: if still uncertain, clarify
        if intent == "answer" and not sufficient:
            intent = "clarify"

        return {
            "intent": intent,
            "questions": data.get("questions", [])[:2],
            "known": known,
            "notes": data.get("notes", "")
        }

    except Exception:
        # Fallback: clarify-first with sensible default questions
        return {
            "intent": "clarify",
            "questions": [
                "which team you're with (Operations, Engineering, Finance, or IT)",
                "if it’s for Wind, Solar, or Battery (and the site/plant if applicable)"
            ],
            "known": {},
            "notes": "router_exception_fallback"
        }


# --- Answer Prompt (only for final answers) ---
SYSTEM_INSTRUCTIONS = (
    "You are Diva, an internal Deriva Energy assistant for charging guidelines. "
    "If the user is just greeting you (like 'hi', 'hello', 'hey', etc.), respond with a simple, friendly greeting and ask how you can help with charging guidelines. Do NOT use bullet points for greetings.\n\n"
    "For all other queries about guidelines, codes, departments, projects, etc., return a markdown bulleted list with exactly these fields:\n"
    "- **Description:**\n- **Account number:**\n- **Location:**\n- **Company ID:**\n- **Project:**\n- **Department:**\n"
    "Use 'N/A' when unavailable. Finish with 1-2 short notes if needed."
)

#"""
# If it's a greeting, greet back (your name is Diva. Made by Deriva Energy.) Else, based on the following retrieved information and the user's query, please provide the most relevant details about charging guidelines.\n\nExtract and present the following in a markdown bulleted list:\n\n- **Description:**\n- **Account number:**\n- **Location:**\n- **Company ID:**\n- **Project:**\n- **Department:**\n\nIf not available, return \"N/A\".\n\nFinish with 1-2 relevant notes if needed.
# """

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# --- Clarification Prompt (for asking questions) ---
CLARIFICATION_INSTRUCTIONS = (
    "You are Diva, an internal Deriva Energy assistant. "
    "Based on the conversation and the questions you need to ask, respond naturally and conversationally. "
    "Ask the clarifying questions in a friendly, helpful manner. Do NOT provide bullet points or structured data yet."
)

clarification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLARIFICATION_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Please ask these clarifying questions: {questions}")
    ]
)

def generate_clarification(user_input: str, questions: List[str]) -> str:
    """Generate a natural clarifying response - simplified for speed"""
    # Use a simple template instead of calling the LLM for speed
    q_text = " ".join(questions[:2])  # Join questions naturally
    return f"I'd be happy to help! To give you the right charging guidelines, could you tell me {q_text}?"
def generate_answer(user_input: str) -> Dict:
    retrieval = retrieve_from_kb(user_input)
    context = retrieval["context"]
    mv = memory.load_memory_variables({})
    messages = answer_prompt.format_messages(context=context, chat_history=mv["chat_history"], input=user_input)
    llm_resp = chat_model.invoke(messages)
    
    # Use the standard LangChain memory methods
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(llm_resp.content)
    
    return {"answer_md": llm_resp.content, "sources": retrieval["sources"]}

# --- Render existing history ---
for m in chat_history_store.messages:
    role = "assistant" if m.type in ("ai", "assistant") else "user"
    with st.chat_message(role):
        st.markdown(m.content)

# --- Chat loop ---
user_input = st.chat_input("Ask about codes, departments, projects, etc.")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1) Route: clarify or answer?
    try:
        decision = route_turn(user_input)
    except:
        # If router fails, default to answer for speed
        decision = {"intent": "answer", "questions": [], "known": {}, "notes": ""}

    if decision["intent"] == "clarify" and decision.get("questions"):
        # Generate a natural clarifying response
        clarifier = generate_clarification(user_input, decision["questions"])
        
        # Save to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(clarifier)

        with st.chat_message("assistant"):
            st.markdown(clarifier)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = generate_answer(user_input)
                    st.markdown(result["answer_md"])
                    
                    # Comment out sources for production, uncomment for dev
                    # if result["sources"]:
                    #     with st.expander("Sources"):
                    #         for i, s in enumerate(result["sources"], 1):
                    #             loc = s.get("location", {})
                    #             score = s.get("score")
                    #             st.markdown(f"- {i}. `{json.dumps(loc)}`" + (f"  (score: {score:.3f})" if score is not None else ""))
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# --- Footer ---
st.divider()
footer = """
<style>
a:link , a:visited{ color: blue; background-color: transparent; text-decoration: underline; }
a:hover, a:active { color: red; background-color: transparent; text-decoration: underline; }
.footer { position: fixed; left:0; bottom:0; width:100%; background-color:white; color:black; text-align:center; }
</style>
<div class="footer">
<p>Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
