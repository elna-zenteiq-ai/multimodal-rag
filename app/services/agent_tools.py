"""Agent-facing tools backed by the RAGService.

Expose useful retrieval functions as LangChain tools so agents can call them.
"""
from typing import List, Optional
import json

from langchain.tools import tool, ToolRuntime

from app.services.agent_context import AgentContext


@tool("rag_search", description="Search documents using the RAG vector store. Returns JSON list of matches.")
def rag_search_tool(query: str, top_k: int = 5, file_ids: Optional[List[str]] = None, conversation_id: Optional[str] = None, runtime: ToolRuntime[AgentContext] | None = None) -> str:
    """Search the vectorstore for documents relevant to `query`.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        file_ids: Optional list of file ids to scope the search.
        conversation_id: Optional conversation id to scope the search.

    Returns:
        JSON string representing a list of matches. Each match contains `text` and `metadata`.
    """
    # Defer import to avoid circular dependency
    from app.services.rag_service import get_rag_service
    
    rag = get_rag_service()
    # If runtime provided, prefer its conversation_id when explicit not passed
    try:
        convo = conversation_id
        if runtime and runtime.context:
            # runtime.context is an AgentContext instance
            if convo is None:
                convo = getattr(runtime.context, "conversation_id", None)

        docs = rag.rag_search(query=query, top_k=top_k, file_ids=file_ids, conversation_id=convo)
    except Exception as e:
        return json.dumps({"error": str(e)})

    results = []
    for d in docs:
        results.append({
            "text": d.page_content,
            "metadata": d.metadata or {},
        })

    return json.dumps(results)
