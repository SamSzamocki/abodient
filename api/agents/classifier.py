import os, json
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

# 1. Initialise Pinecone v3 client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("urgency-search")

# 2. Prepare models
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o-mini")      # inexpensive GPT-4 tier


SYSTEM_PROMPT = (
    "You are a property-management assistant. "
    'Return JSON like {"urgency":"low|medium|high","responsibility":"tenant|landlord|unknown"}.'
)

def classify(text: str) -> dict:
    """Return urgency & responsibility for a tenant message."""
    # -- fetch similar past cases from vector store
    embedding = embedder.embed_query(text)
    matches = index.query(vector=embedding, top_k=3, include_metadata=True)
    snippets = "\n".join(m["metadata"]["text"] for m in matches["matches"])

    # -- ask the LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Tenant text: {text}\nSimilar cases:\n{snippets}"),
    ]
    reply_json = llm(messages).content

    # -- convert to Python dict and return
    return json.loads(reply_json)
