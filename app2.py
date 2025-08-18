# app.py
import os
import re
import io
import csv
import json
import uuid
import time
from typing import List, Dict, Any, Tuple, Optional

import boto3
import streamlit as st

# =========================
# Config
# =========================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "YBW1J8NMTI")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# =========================
# UI Setup
# =========================
st.set_page_config(
    page_title="Diva (General Policy Assistant)",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Diva is for internal use. It may contain errors.")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Persist known filters & pending questions across turns (deterministic, not LLM)
st.session_state.setdefault("known_filters", {})       # { field_name: chosen_value }
st.session_state.setdefault("pending_fields", [])      # [field_name...]
st.session_state.setdefault("options_map", {})         # { field_name: [values...] }
st.session_state.setdefault("last_options_snapshot", {})  # for debug/inspection

# =========================
# DynamoDB Chat History (uses existing table schema)
# PK: session_id (S), SK: message_timestamp (S)
# =========================
class CustomDynamoDBChatHistory:
    def __init__(self, table_name: str, session_id: str):
        self.table_name = table_name
        self.session_id = session_id
        self.table = dynamodb.Table(table_name)

    def messages(self) -> List[Dict[str, str]]:
        try:
            from boto3.dynamodb.conditions import Key
            resp = self.table.query(
                KeyConditionExpression=Key("session_id").eq(self.session_id),
                ScanIndexForward=True
            )
            # Return as [{"role":"user"/"assistant","content": "..."}]
            out = []
            for it in resp.get("Items", []):
                role = "assistant" if it.get("message_type") in ("ai", "assistant") else "user"
                out.append({"role": role, "content": it.get("content", "")})
            return out
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
            return []

    def add(self, role: str, content: str) -> None:
        try:
            msg_type = "human" if role == "user" else "ai"
            self.table.put_item(
                Item={
                    "session_id": self.session_id,
                    "message_timestamp": str(int(time.time() * 1000)),
                    "message_type": msg_type,
                    "content": content
                }
            )
        except Exception as e:
            st.error(f"Failed to save message: {e}")

    def clear(self):
        try:
            from boto3.dynamodb.conditions import Key
            resp = self.table.query(
                KeyConditionExpression=Key("session_id").eq(self.session_id)
            )
            with self.table.batch_writer() as batch:
                for it in resp.get("Items", []):
                    batch.delete_item(
                        Key={
                            "session_id": it["session_id"],
                            "message_timestamp": str(it["message_timestamp"])
                        }
                    )
        except Exception as e:
            st.error(f"Failed to clear history: {e}")

history = CustomDynamoDBChatHistory(DDB_TABLE_NAME, st.session_state["session_id"])

def reset_history():
    st.session_state["known_filters"] = {}
    st.session_state["pending_fields"] = []
    st.session_state["options_map"] = {}
    st.session_state["last_options_snapshot"] = {}
    try:
        history.clear()
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

with st.sidebar.expander("🧹 Tools", expanded=True):
    if st.button("🗑️ Clear Chat"):
        reset_history()
        st.rerun()

st.markdown("<h1 style='text-align:center;'>⚡ Meet Diva</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>General, clarify-first assistant powered by your policy KB.</p>", unsafe_allow_html=True)

# =========================
# Knowledge Base Retrieval
# =========================
def retrieve_from_kb(query: str, max_results: int = 6) -> Dict[str, Any]:
    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": max_results}},
    )
    chunks, sources = [], []
    for item in resp.get("retrievalResults", []):
        txt = item.get("content", {}).get("text", "")
        loc = item.get("location", {})
        score = item.get("score")
        if txt:
            chunks.append(txt.strip())
        if loc:
            sources.append({"location": loc, "score": score})
    return {
        "context": "\n\n---\n\n".join(chunks),
        "sources": sources
    }

# =========================
# Generic Context Parsing
# (no domain-specific logic)
# =========================

CANON_KEYS = {
    # Canonical 6-field output keys (lowercase)
    "description": {"description", "desc"},
    "account number": {"account", "account number", "acct", "acct #", "acct#", "gl account"},
    "location": {"location", "site"},
    "company id": {"company id", "companyid", "company_id"},
    "project": {"project", "project code", "proj", "proj code"},
    "department": {"department", "dept", "dept code", "department code"},
}

# Other potentially discriminative columns we can ask about (generic)
# (We don't special-case any domain; these are just common tabular attributes)
GENERIC_DIMENSION_HINTS = {
    "category", "type", "activity", "subtype", "class", "group",
    "cost center", "costcenter", "program", "function"
}

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _best_header_map(headers: List[str]) -> Dict[str, str]:
    """
    Map raw headers to canonical names where possible.
    Returns {canonical_or_raw: raw_header}
    """
    norm_headers = [_normalize(h) for h in headers]
    mapping: Dict[str, str] = {}
    used = set()
    for canon, aliases in CANON_KEYS.items():
        for i, nh in enumerate(norm_headers):
            if nh in aliases and headers[i] not in used:
                mapping[canon] = headers[i]
                used.add(headers[i])
                break
    # add generic candidate dimensions (kept as their raw header)
    for i, nh in enumerate(norm_headers):
        if headers[i] in used:
            continue
        if nh in GENERIC_DIMENSION_HINTS:
            mapping[headers[i]] = headers[i]
    # also include any remaining headers as potential filters (kept raw)
    for h in headers:
        if h not in used and h not in mapping.values():
            mapping[h] = h
    return mapping

def _sniff_delimiter(line: str) -> str:
    # Try common delimiters
    for delim in [",", "|", "\t", ";"]:
        if line.count(delim) >= 2:
            return delim
    return ","  # default

def _find_table_blocks(text: str) -> List[str]:
    """
    Heuristic: find blocks that look like tables (header row with several separators)
    Returns list of blocks (as text).
    """
    lines = [l for l in text.splitlines() if l.strip() != ""]
    blocks = []
    i = 0
    while i < len(lines):
        if re.search(r"(,|\||;|\t)", lines[i]) and re.search(r"[A-Za-z]", lines[i]):
            delim = _sniff_delimiter(lines[i])
            # grow block while the delimiter pattern continues
            j = i
            rows = []
            while j < len(lines) and lines[j].count(delim) >= 1:
                rows.append(lines[j])
                j += 1
            if len(rows) >= 2:
                blocks.append("\n".join(rows))
            i = j
        else:
            i += 1
    return blocks

def parse_tables_from_context(context: str) -> List[List[Dict[str, str]]]:
    """
    Parse all table-like blocks into list of tables; each table is a list of row dicts.
    """
    tables: List[List[Dict[str, str]]] = []
    for block in _find_table_blocks(context):
        try:
            # try CSV sniffer
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(block.splitlines()[0])
            reader = csv.reader(io.StringIO(block), dialect)
        except Exception:
            delim = _sniff_delimiter(block.splitlines()[0])
            reader = csv.reader(io.StringIO(block), delimiter=delim)

        rows = list(reader)
        if not rows:
            continue
        headers = [h.strip() for h in rows[0]]
        if not headers or len(headers) < 2:
            continue

        # normalize duplicate headers
        seen = {}
        clean_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                clean_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 1
                clean_headers.append(h)

        table: List[Dict[str, str]] = []
        for r in rows[1:]:
            if len(r) != len(clean_headers):
                # pad or skip
                if len(r) < len(clean_headers):
                    r = r + [""] * (len(clean_headers) - len(r))
                else:
                    r = r[:len(clean_headers)]
            row = {clean_headers[i]: r[i].strip() for i in range(len(clean_headers))}
            table.append(row)
        tables.append(table)
    return tables

def unique_values(table: List[Dict[str, str]], col: str, limit: int = 50) -> List[str]:
    vals = []
    seen = set()
    for row in table:
        v = row.get(col, "").strip()
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
        if len(vals) >= limit:
            break
    return vals

def select_discriminators(table: List[Dict[str, str]], header_map: Dict[str, str], known: Dict[str, str]) -> List[str]:
    """
    Choose up to 2 columns to ask about, prioritizing:
    - not already known
    - high uniqueness (more options = more informative)
    - canonical columns first (project/department/location/category-ish) when present
    """
    # Rank columns by number of unique values
    candidates = []
    for canon_or_raw, raw in header_map.items():
        if canon_or_raw in known:
            continue
        uv = unique_values(table, raw, limit=1000)
        if len(uv) > 1:
            candidates.append((canon_or_raw, len(uv)))

    # Weighted preference for canonical 6-field columns and generic dimension hints
    def weight(name: str) -> int:
        n = _normalize(name)
        if n in CANON_KEYS:
            return 3
        if n in GENERIC_DIMENSION_HINTS:
            return 2
        return 1

    candidates.sort(key=lambda x: (weight(x[0]), x[1]), reverse=True)
    return [c for c, _ in candidates[:2]]

def filter_rows(table: List[Dict[str, str]], header_map: Dict[str, str], known: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Filter rows by known filters (matching on the raw columns).
    """
    def match(row: Dict[str, str]) -> bool:
        for k, v in known.items():
            raw = header_map.get(k) or header_map.get(k.lower()) or k
            rv = row.get(raw, "")
            if not rv:
                return False
            # case-insensitive containment or exact match for numeric-like
            if re.fullmatch(r"\d+(\.\d+)?", v.strip()):
                if rv.strip() != v.strip():
                    return False
            else:
                if _normalize(v) not in _normalize(rv):
                    return False
        return True

    return [r for r in table if match(r)]

def row_to_six_fields(row: Dict[str, str], header_map: Dict[str, str]) -> Dict[str, str]:
    """
    Create the 6 required fields from whatever headers we have.
    Unknowns become 'N/A'.
    """
    def g(key: str) -> str:
        raw = header_map.get(key)
        return (row.get(raw, "").strip() if raw else "") or "N/A"

    # Try sensible fallbacks for description
    desc = g("description")
    if desc == "N/A":
        for alt in ["activity", "expense type", "type", "category"]:
            raw = header_map.get(alt)
            if raw:
                v = row.get(raw, "").strip()
                if v:
                    desc = v
                    break

    return {
        "Description": desc,
        "Account number": g("account number"),
        "Location": g("location"),
        "Company ID": g("company id"),
        "Project": g("project"),
        "Department": g("department"),
    }

# =========================
# Clarify-First Router (deterministic)
# =========================
def plan_questions(user_input: str, context: str, known: Dict[str, str]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    """
    Analyze retrieved context, parse tables, pick up to 2 discriminators,
    and build options for each. Returns (questions, options_map, parse_snapshot)
    """
    tables = parse_tables_from_context(context)
    snapshot = {"tables_found": len(tables), "per_table": []}
    questions: List[str] = []
    options_map: Dict[str, List[str]] = {}

    # Aggregate columns/values across tables
    for t in tables:
        if not t:
            continue
        headers = list(t[0].keys())
        header_map = _best_header_map(headers)

        # Decide discriminators for this table
        fields = select_discriminators(t, header_map, known)
        # Build options for those fields
        for f in fields:
            raw = header_map.get(f, f)
            opts = unique_values(t, raw, limit=30)
            if opts:
                # keep short, distinct options
                options_map.setdefault(f, [])
                for o in opts:
                    if o not in options_map[f]:
                        options_map[f].append(o)

        snapshot["per_table"].append({
            "headers": headers,
            "mapped": header_map,
            "discriminators": fields,
        })

    # Compose up to two distinct fields with options
    # Prefer ones that actually have options and aren't already known
    prioritized = [f for f, opts in options_map.items() if opts and f not in known]
    prioritized = prioritized[:2]

    for f in prioritized:
        # build question with a few options inline
        sample = ", ".join(options_map[f][:6])
        label = f.title()
        questions.append(f"Which **{label}**? (e.g., {sample})")

    return questions[:2], options_map, snapshot

def try_resolve_answer(context: str, known: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    If we can deterministically select a single row from any parsed table with current known filters,
    return the 6-field dict; else None.
    """
    tables = parse_tables_from_context(context)
    for t in tables:
        if not t:
            continue
        headers = list(t[0].keys())
        header_map = _best_header_map(headers)
        matches = filter_rows(t, header_map, known)
        if len(matches) == 1:
            return row_to_six_fields(matches[0], header_map)
    return None

# =========================
# Rendering helpers
# =========================
def render_history():
    for m in history.messages():
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def render_answer_block(payload: Dict[str, str]):
    lines = [
        f"- **Description:** {payload.get('Description','N/A')}",
        f"- **Account number:** {payload.get('Account number','N/A')}",
        f"- **Location:** {payload.get('Location','N/A')}",
        f"- **Company ID:** {payload.get('Company ID','N/A')}",
        f"- **Project:** {payload.get('Project','N/A')}",
        f"- **Department:** {payload.get('Department','N/A')}",
    ]
    st.markdown("\n".join(lines))

def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    with st.expander("Sources"):
        for i, s in enumerate(sources, 1):
            loc = s.get("location", {})
            score = s.get("score")
            st.markdown(f"- {i}. `{json.dumps(loc)}`" + (f"  (score: {score:.3f})" if score is not None else ""))

# =========================
# Chat loop
# =========================
render_history()

user_input = st.chat_input("Ask your policy question (I’ll clarify first).")
if user_input:
    # record user
    history.add("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    text_norm = _normalize(user_input)

    # If greeting, respond simply
    if any(text_norm.startswith(g) for g in ["hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"]):
        reply = "Hi! How can I help with charging/policy details today?"
        history.add("assistant", reply)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.stop()

    # Try to map user's reply to any pending field options (deterministic capture)
    # This lets users answer with just the value, e.g., "8032" or "Lodging"
    if st.session_state["pending_fields"] and st.session_state["options_map"]:
        for field in list(st.session_state["pending_fields"]):
            opts = st.session_state["options_map"].get(field, [])
            # Try exact or case-insensitive contains match
            picked = None
            for o in opts:
                if _normalize(o) == _normalize(user_input) or _normalize(o) in _normalize(user_input):
                    picked = o
                    break
            if picked:
                st.session_state["known_filters"][field] = picked
                st.session_state["pending_fields"].remove(field)

    # Retrieve KB context for this turn
    retrieval = retrieve_from_kb(user_input)
    context = retrieval["context"]

    # Attempt deterministic answer with what we know now
    six = try_resolve_answer(context, st.session_state["known_filters"])

    with st.chat_message("assistant"):
        if six is not None:
            # Deterministic final answer
            render_answer_block(six)
            history.add("assistant", "\n".join([
                f"- **Description:** {six['Description']}",
                f"- **Account number:** {six['Account number']}",
                f"- **Location:** {six['Location']}",
                f"- **Company ID:** {six['Company ID']}",
                f"- **Project:** {six['Project']}",
                f"- **Department:** {six['Department']}",
            ]))
            # Optional: show sources
            render_sources(retrieval["sources"])
        else:
            # Plan clarifying questions from this context
            questions, options_map, snapshot = plan_questions(user_input, context, st.session_state["known_filters"])
            st.session_state["options_map"] = options_map
            st.session_state["last_options_snapshot"] = snapshot

            if questions:
                # Save which fields we're asking so we can capture the next reply
                st.session_state["pending_fields"] = [q.split("**")[1].split("**")[0].lower() for q in questions if "**" in q]  # extract field labels
                # Friendly, concise clarifier
                clarifier = "To get you the exact codes, could you tell me:\n\n" + "\n".join([f"• {q}" for q in questions])
                st.markdown(clarifier)
                history.add("assistant", clarifier)

                # Provide an expander with options (from the policy text)
                if options_map:
                    with st.expander("Options I found in the policy (pick & reply with one):"):
                        for field, opts in options_map.items():
                            if not opts:
                                continue
                            st.markdown(f"**{field.title()}**")
                            st.markdown(", ".join([f"`{o}`" for o in opts[:50]]))
                # Optional: sources for verification
                render_sources(retrieval["sources"])
            else:
                # No clear questions to ask (no parseable table, or already too narrow)
                # Return safe 6-field skeleton with N/A and a note
                fallback = {
                    "Description": "N/A",
                    "Account number": "N/A",
                    "Location": "N/A",
                    "Company ID": "N/A",
                    "Project": "N/A",
                    "Department": "N/A",
                }
                st.markdown("_I couldn’t find a clear table to resolve this directly. Here’s a skeleton—reply with any specific field (e.g., department, project, location) and I’ll drill down:_")
                render_answer_block(fallback)
                history.add("assistant", "\n".join([
                    f"- **Description:** {fallback['Description']}",
                    f"- **Account number:** {fallback['Account number']}",
                    f"- **Location:** {fallback['Location']}",
                    f"- **Company ID:** {fallback['Company ID']}",
                    f"- **Project:** {fallback['Project']}",
                    f"- **Department:** {fallback['Department']}",
                ]))
                render_sources(retrieval["sources"])

# =========================
# Footer
# =========================
st.divider()
st.caption("Diva is general-purpose. It mines options from whatever the KB returns—no hard-coded departments or categories. If a value isn’t in the policy text I retrieved, I won’t guess.")
