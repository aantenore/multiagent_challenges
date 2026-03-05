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

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    team_name = os.getenv("TEAM_NAME", "A(CC)I-Tua")
    return f"{team_name}-{ulid.new().str}"

_CURRENT_SESSION_ID = generate_session_id()

def set_current_session_id(session_id: str):
    global _CURRENT_SESSION_ID
    _CURRENT_SESSION_ID = session_id

def get_current_session_id():
    return _CURRENT_SESSION_ID

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
