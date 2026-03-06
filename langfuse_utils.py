import os
import ulid
from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

import threading

_local = threading.local()

def generate_session_id(prefix: str | None = None):
    """Generate a unique session ID, optionally with a prefix."""
    team_name = os.getenv("TEAM_NAME", "A(CC)I-Tua")
    suffix = ulid.new().str
    if prefix:
        return f"{prefix}-{suffix}"
    return f"{team_name}-{suffix}"

def set_current_session_id(session_id: str):
    _local.session_id = session_id

def get_current_session_id():
    if not hasattr(_local, "session_id"):
        _local.session_id = generate_session_id()
    return _local.session_id

def invoke_langchain(model, system_message, prompt, langfuse_handler):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content

@observe()
def run_llm_call(session_id, model, system_message, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Update trace with session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # Create Langfuse callback handler for automatic generation tracking
    # The handler will attach to the current trace created by @observe()
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, system_message, prompt, langfuse_handler)

    return response

print("✓ Langfuse initialized successfully")
print(f"✓ Public key: {os.getenv('LANGFUSE_PUBLIC_KEY', 'Not set')[:20]}...")
print("✓ Helper functions ready: generate_session_id(), invoke_langchain(), run_llm_call()")
