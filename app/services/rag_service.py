"""RAG service for retrieval and generation with NVIDIA GPT-OSS-120B."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_agent
import asyncio

from app.services.agent_tools import rag_search_tool
from app.services.agent_context import AgentContext

from app.config import get_settings
from app.models import QueryResponse, SourceDocument
from app.services.milvus_service import get_milvus_service


class RAGService:
    """Service for RAG pipeline with NVIDIA GPT-OSS-120B."""

    SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided context and also acts as
an intent classifier and router for retrieval vs. direct-answer behavior.

Rules (follow exactly):
1) INTENT: Determine the user's intent and choose one label: 
    - HIGH_LEVEL: overview, summarise, compare, or ask for conceptual synthesis that can often be
      answered from document summaries.
    - FACTUAL: specific factual lookup, exact numbers, citations, or requests that require precise
      retrieval from source documents.
    - AMBIGUOUS: the user's request is underspecified and requires a clarification question.

2) DECIDE ROUTING: Based on intent, decide whether the LLM can answer from the given summaries
    in the context, or whether a retrieval tool must be invoked. Do not attempt retrieval yourself.
    - If intent=HIGH_LEVEL and the provided summaries appear sufficient, answer directly using only
      the summaries/context.
    - If intent=FACTUAL, do NOT rely on summaries; indicate that retrieval is required.
    - If intent=AMBIGUOUS, ask one concise clarifying question and do not attempt retrieval or answer.

3) OUTPUT FORMAT: Start your reply with a single JSON object on the first line (no surrounding text)
    with these keys: `intent` (one of HIGH_LEVEL/FACTUAL/AMBIGUOUS), `used_summaries` (true|false),
    `clarify` (true|false), and `tool_query` (string or null) which is a short search query suggestion
    if retrieval is required. After that JSON line, provide the human-readable answer (if any), using only
    the allowed context.

4) SOURCES: When you include factual claims from context, cite the source by `file_id` and filename
    present in the context metadata (e.g., "(file: <file_id>, <filename>)").

5) CONSTRAINTS: Use ONLY the information from the provided `context` unless you explicitly signal
    that a retrieval tool should be used (via `tool_query`). If unsure, prefer being conservative and
    request retrieval or clarification.

Context:
{context}
"""

    def __init__(self):
        """Initialize RAG service with NVIDIA LLM."""
        self.settings = get_settings()
        self.milvus = get_milvus_service()

        # Initialize NVIDIA ChatNVIDIA
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=self.settings.nvidia_temperature,
            top_p=self.settings.nvidia_top_p,
            max_tokens=self.settings.nvidia_max_tokens,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # Create an agent that can call `rag_search` as a tool.
        # We create the agent without a fixed system prompt here and pass formatted context at invocation time.
        try:
            # register AgentContext so tools receive runtime.context
            self.agent = create_agent(self.llm, tools=[rag_search_tool], context_schema=AgentContext)
        except Exception:
            self.agent = None

    async def agent_query(self, question: str, file_ids: list[str] | None = None, conversation_id: str | None = None, top_k: int = 5) -> QueryResponse:
        """Run the LLM as an agent with access to `rag_search` tool.

        This method collects summary-context (if available), formats the system prompt with
        that context, and runs the agent. The agent may call `rag_search` automatically.
        """
        # gather summaries as context if available
        context_docs: list[Document] = []
        if self.settings.store_summaries_in_milvus:
            candidates = self.milvus.search(query=question, top_k=self.settings.max_summary_injection * 2)
            for d in candidates:
                md = d.metadata or {}
                if md.get("is_summary"):
                    if file_ids:
                        fid = md.get("file_id")
                        if fid and fid in file_ids:
                            context_docs.append(d)
                    elif conversation_id:
                        cid = md.get("conversation_id")
                        if cid and cid == conversation_id:
                            context_docs.append(d)
                    else:
                        context_docs.append(d)
                if len(context_docs) >= self.settings.max_summary_injection:
                    break

        context = self._format_context(context_docs) if context_docs else ""

        # Ensure we have an agent
        if self.agent is None:
            # fallback to normal aquery
            return await self.aquery(question=question, top_k=top_k)

        # Prepare messages: system prompt with context, then user question
        system_message = self.SYSTEM_PROMPT.format(context=context)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]

        # Run agent in thread to avoid blocking event loop; pass AgentContext as runtime context
        def run_agent():
            return self.agent.invoke({"messages": messages}, context=AgentContext(user_id=None, conversation_id=conversation_id))

        result = await asyncio.to_thread(run_agent)

        # extract text
        text = result.content if hasattr(result, "content") else str(result)

        # After the agent runs, perform a scoped retrieval to build human-friendly sources
        docs = self.rag_search(question, top_k=top_k, file_ids=file_ids, conversation_id=conversation_id)
        sources = self._format_sources(docs)

        # Ensure no internal ids are exposed in the sources (SourceDocument doesn't include file_id)
        return QueryResponse(answer=text, sources=sources)

    def rag_search(self, query: str, top_k: int = 5, file_ids: list[str] | None = None, conversation_id: str | None = None) -> list[Document]:
        """Perform a similarity search and optionally scope by file or conversation.

        Returns a list of LangChain `Document` objects. Filtering is best-effort based on
        metadata keys (`file_id`, `conversation_id`).
        """
        # ask Milvus for extra results and filter locally for scoping
        raw_docs = self.milvus.search(query=query, top_k=max(top_k, 20))
        if not raw_docs:
            return []

        def matches_scope(doc: Document) -> bool:
            md = doc.metadata or {}
            if file_ids:
                fid = md.get("file_id") or md.get("fileid") or md.get("file")
                return fid in file_ids if fid else False
            if conversation_id:
                cid = md.get("conversation_id") or md.get("convo_id")
                return cid == conversation_id if cid else False
            return True

        filtered = [d for d in raw_docs if matches_scope(d)]
        return filtered[:top_k]

    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        """Execute RAG query and return answer with sources.

        Args:
            question: User's question
            top_k: Number of sources to retrieve

        Returns:
            QueryResponse with answer and source documents
        """
        # Retrieve relevant documents
        docs = self.milvus.search(query=question, top_k=top_k)

        # Format context with source info
        context = self._format_context(docs)

        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        # Extract answer text
        answer = response.content if hasattr(response, "content") else str(response)

        # Format source documents
        sources = self._format_sources(docs)

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    async def aquery(self, question: str, top_k: int = 5, conversation_id: str | None = None) -> QueryResponse:
        """Async version of query.

        Args:
            question: User's question
            top_k: Number of sources to retrieve
            conversation_id: Optional conversation ID for scoped retrieval

        Returns:
            QueryResponse with answer and source documents
        """
        # Retrieve relevant documents
        docs = self.milvus.search(query=question, top_k=top_k)

        # Format context with source info
        context = self._format_context(docs)

        # Generate answer asynchronously
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "question": question,
        })

        # Extract answer text
        raw_answer = response.content if hasattr(response, "content") else str(response)
        
        # Parse the answer: skip the JSON header and get the human-readable part
        answer = raw_answer
        try:
            import json
            # Find the first JSON object
            start_idx = raw_answer.find('{')
            if start_idx != -1:
                # Find the matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(raw_answer)):
                    if raw_answer[i] == '{':
                        brace_count += 1
                    elif raw_answer[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                # Skip the JSON and get the rest as the answer
                if end_idx < len(raw_answer):
                    answer = raw_answer[end_idx:].strip()
                    if not answer:
                        answer = raw_answer
        except Exception:
            # If parsing fails, just use the raw answer
            pass

        # Format source documents
        sources = self._format_sources(docs)

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    def _format_context(self, docs: list[Document]) -> str:
        """Format retrieved documents into context string with source info.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        contexts = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata

            # Build source citation
            source_parts = []
            filename = metadata.get("filename", "Unknown")
            source_parts.append(f"Source: {filename}")

            page_numbers_raw = metadata.get("page_numbers", [])
            if isinstance(page_numbers_raw, str):
                page_numbers = [int(p) for p in page_numbers_raw.split(",") if p.strip()]
            else:
                page_numbers = page_numbers_raw

            if page_numbers:
                pages_str = ", ".join(str(p) for p in page_numbers)
                source_parts.append(f"Page(s): {pages_str}")

            heading = metadata.get("heading")
            if heading:
                source_parts.append(f"Section: {heading}")

            # Combine text with source info
            source_info = " | ".join(source_parts)
            contexts.append(f"[{i}] {doc.page_content}\n--- {source_info} ---")

        return "\n\n".join(contexts)

    def _format_sources(self, docs: list[Document]) -> list[SourceDocument]:
        """Format retrieved documents into SourceDocument objects.

        Args:
            docs: List of retrieved documents

        Returns:
            List of SourceDocument objects
        """
        sources = []

        for doc in docs:
            metadata = doc.metadata

            # Parse page numbers from string if needed
            page_numbers_raw = metadata.get("page_numbers", [])
            if isinstance(page_numbers_raw, str):
                page_numbers = [int(p) for p in page_numbers_raw.split(",") if p.strip()]
            else:
                page_numbers = page_numbers_raw

            source = SourceDocument(
                text=doc.page_content,
                filename=metadata.get("filename", "Unknown"),
                page_numbers=page_numbers,
                heading=metadata.get("heading"),
            )
            sources.append(source)

        return sources


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
