"""Agent-based RAG service using LangChain agent framework with short-term memory.

Architecture
------------
- A *single* LangGraph agent is created per service instance, backed by a
  PostgresSaver checkpointer so that every ``conversation_id`` maps to its
  own persistent *thread*.
- On the **first query** in a conversation the file summaries are injected as a
  system message so the agent knows what documents are available.  On all
  subsequent queries the agent already has those summaries in its short-term
  memory and does **not** re-inject them.
- There is **no separate intent-classification step**.  The agent's system
  prompt instructs it to judge autonomously whether it can answer from the
  summaries in its memory or whether it needs to call the ``rag_search``
  tool to retrieve detailed document chunks.
"""

import logging
from typing import Optional

from langchain import agents
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg

from app.config import get_settings
from app.database import get_db_service
from app.services.milvus_service import get_milvus_service
from app.services.summary_search import get_summary_search_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — inspired by the deep-research agent style
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a document analysis assistant with access to the user's uploaded files.

<Task>
Answer the user's questions by combining two information sources:
1. **File summaries** — short overviews of each document that have been provided
   to you at the start of the conversation (in your memory / message history).
2. **rag_search tool** — performs similarity search over the full document
   chunks stored in the vector database and returns the most relevant passages.
</Task>

<Available Tools>
1. **rag_search**: Search uploaded documents for relevant passages.
   - Accepts a `query` (str) and optional `top_k` (int, default 5).
   - Returns formatted text with source metadata (filename, page, heading).
</Available Tools>

<Instructions>
Think like a thorough research analyst. Follow these steps for every query:

1. **Read the question carefully** — understand exactly what information the
   user needs.
2. **Decide whether the summaries are sufficient** — if the user asks for a
   very high-level overview (e.g. "what is this document about?") and the
   summaries already contain the answer, you may respond directly.
3. **For anything that needs detail, use rag_search** — specific facts, lists,
   guidelines, comparisons, quotes, data points, section contents, etc.
   *When in doubt, search.* It is much better to search and confirm than to
   guess from a short summary.
4. **Synthesize a clear, well-structured answer** from the retrieved passages.
5. **Do NOT embed citations** like "(p. 3)" or "(section 1.2)" in the answer
   text. Source metadata (filenames, pages, headings) is already provided
   separately in the API response. Just write a clean answer.

IMPORTANT RULES:
- Do NOT fabricate information.  If the documents do not contain the answer,
  say so honestly.
- Do NOT repeat the raw search output verbatim — rephrase and organise it.
- If multiple documents are relevant, compare and consolidate the information.
- Keep responses focused, comprehensive, and professional.
</Instructions>

<Response Format>
- Write **one or two short, dense paragraphs** of plain prose — similar in
  style to an executive summary or an abstract.
- No headings, no tables, no bullet lists, no checklists, no markdown
  formatting tricks. Just clean sentences that flow naturally.
- You may bold a key term on first mention if it helps clarity, but avoid
  heavy formatting.
- Do NOT include inline citations, page numbers, or section references in
  parentheses. The system already returns structured source metadata
  alongside your answer — the user will see it there.
- Aim for roughly 3–6 sentences. Be thorough but never verbose.
</Response Format>

<Hard Limits>
- Maximum 3 rag_search calls per user query (avoid excessive searching).
- Stop searching once you can answer the question confidently.
- If the first search returns what you need, do not search again.
</Hard Limits>
"""


class AgentRAGService:
    """Agent-based RAG with PostgreSQL-backed short-term memory."""

    def __init__(self):
        self.settings = get_settings()
        self.milvus = get_milvus_service()
        self.db_service = get_db_service()
        self.summary_search = get_summary_search_service()

        # LLM
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=self.settings.nvidia_temperature,
            top_p=self.settings.nvidia_top_p,
            max_completion_tokens=self.settings.nvidia_max_tokens,
        )

        # Async checkpointer — lazily initialised on first query because
        # we need an async psycopg connection which can only be created
        # inside a running event loop.
        self._checkpointer: AsyncPostgresSaver | None = None
        self._pg_conn: psycopg.AsyncConnection | None = None

    # ------------------------------------------------------------------
    # Lazy async checkpointer
    # ------------------------------------------------------------------
    async def _ensure_checkpointer(self) -> AsyncPostgresSaver:
        """Lazily create the async PostgresSaver on first use."""
        if self._checkpointer is not None:
            return self._checkpointer

        self._pg_conn = await psycopg.AsyncConnection.connect(
            self.settings.database_url,
            autocommit=True,
            prepare_threshold=0,
        )
        self._checkpointer = AsyncPostgresSaver(self._pg_conn)
        await self._checkpointer.setup()
        logger.info("AsyncPostgresSaver initialized successfully")
        return self._checkpointer

    # ------------------------------------------------------------------
    # RAG Tool factory
    # ------------------------------------------------------------------
    def _create_rag_tool(
        self,
        file_ids: list[str],
        chunk_ids: list[str] | None = None,
        retrieved_docs: list | None = None,
    ) -> StructuredTool:
        """Create a rag_search tool scoped to the conversation's chunks."""
        _retrieved = retrieved_docs if retrieved_docs is not None else []

        def rag_search_scoped(query: str, top_k: int = 5) -> str:
            """Search documents scoped to specific files via their chunk IDs."""
            try:
                if chunk_ids:
                    results = self.milvus.search_by_chunk_ids(
                        query=query, chunk_ids=chunk_ids, top_k=top_k
                    )
                else:
                    results = self.milvus.search(query=query, top_k=top_k)
                    if file_ids:
                        results = [
                            d for d in results if d.metadata.get("file_id") in file_ids
                        ]

                if not results:
                    logger.info("rag_search_scoped: no results found")
                    return "No relevant information found in the specified documents."

                _retrieved.extend(results)
                logger.info(
                    f"rag_search_scoped: {len(results)} results found, "
                    f"_retrieved now has {len(_retrieved)} items"
                )

                from app.services.rag_tool import _format_search_results
                return _format_search_results(results, file_ids)

            except Exception as e:
                logger.error(f"Error in RAG search: {e}")
                return f"Error searching documents: {str(e)}"

        return StructuredTool.from_function(
            func=rag_search_scoped,
            name="rag_search",
            description=(
                "Search through the uploaded documents to find relevant passages. "
                "Use this tool whenever you need specific details, facts, guidelines, "
                "lists, or any information beyond the brief summaries in your memory."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_first_query(self, conversation_id: str) -> bool:
        """Return True if no user messages exist yet for this conversation."""
        msgs = self.db_service.get_conversation_messages(conversation_id)
        return len(msgs) == 0

    def _build_summary_context(self, conversation_id: str) -> str:
        """Build a summary context string for the conversation's documents."""
        summaries = self.summary_search.get_conversation_summaries(conversation_id)
        if not summaries:
            return ""
        parts = []
        for fid, summary in summaries:
            file_obj = self.db_service.get_file(fid)
            fname = file_obj.filename if file_obj else fid
            parts.append(f"**{fname}** (id: {fid}):\n{summary}")
        return (
            "The following documents have been uploaded to this conversation. "
            "Their short summaries are provided below for orientation. "
            "Use the rag_search tool to retrieve more detailed content when "
            "needed.\n\n" + "\n\n".join(parts)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def query(
        self,
        query: str,
        conversation_id: str,
        file_ids: Optional[list[str]] = None,
        chunk_ids: Optional[list[str]] = None,
    ) -> dict:
        """Run a user query through the agent.

        Args:
            query: User question.
            conversation_id: Conversation / thread ID.
            file_ids: File IDs in the conversation.
            chunk_ids: Milvus chunk IDs for scoped retrieval.

        Returns:
            Dict with keys: answer, sources, used_rag.
        """
        # Resolve files / chunks if not provided
        if file_ids is None:
            files = self.db_service.get_conversation_files(conversation_id)
            file_ids = [f.file_id for f in files if f.summary]
        if chunk_ids is None and file_ids:
            chunk_ids = self.db_service.get_chunk_ids_for_files(file_ids)

        logger.info(
            f"Processing query: {query[:100]}... | "
            f"conversation={conversation_id} | files={len(file_ids)}"
        )

        # Shared list to capture raw docs returned by the tool
        retrieved_docs: list = []

        # Build tools scoped to this conversation's chunks
        tools = [
            self._create_rag_tool(file_ids, chunk_ids=chunk_ids, retrieved_docs=retrieved_docs)
        ]

        # Create a new agent graph with the checkpointer.
        # The agent is re-created per call so tools can be scoped to the
        # current conversation's files / chunks; the *state* (memory) is
        # persisted via the checkpointer keyed by thread_id.
        checkpointer = await self._ensure_checkpointer()
        agent = agents.create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

        # Build the messages to send in *this* invocation.
        messages: list = []

        # On first query inject document summaries so the agent has context.
        if self._is_first_query(conversation_id):
            summary_ctx = self._build_summary_context(conversation_id)
            if summary_ctx:
                messages.append(SystemMessage(content=summary_ctx))

        messages.append(HumanMessage(content=query))

        # Invoke agent — thread_id = conversation_id
        config = {"configurable": {"thread_id": conversation_id}}
        result = await agent.ainvoke({"messages": messages}, config)

        # Extract answer from the last AI message
        answer = "Unable to generate answer."
        output_messages = result.get("messages", [])
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        # used_rag is based on whether the tool was actually called in
        # THIS invocation (i.e. retrieved_docs was populated via the
        # closure), NOT on historical ToolMessages replayed by the
        # checkpointer.
        used_rag = len(retrieved_docs) > 0

        logger.info(
            f"Agent finished: total_msgs={len(output_messages)}, "
            f"retrieved_docs={len(retrieved_docs)}, used_rag={used_rag}"
        )

        # Sort by relevance (lowest L2 distance first) and cap at 5
        retrieved_docs.sort(
            key=lambda d: d.metadata.get("_distance", 999)
        )

        # Build source citations from captured documents
        _MAX_SOURCES = 5
        sources = []
        seen: set = set()
        for doc in retrieved_docs:
            if len(sources) >= _MAX_SOURCES:
                break
            meta = doc.metadata
            key = (meta.get("filename", ""), doc.page_content[:100])
            if key in seen:
                continue
            seen.add(key)

            # Normalise page_numbers — Milvus stores VARCHAR so it may
            # come back as a plain string like "3" or a JSON list "[1,2]".
            raw_pn = meta.get("page_numbers", [])
            if isinstance(raw_pn, str):
                import json as _json
                try:
                    raw_pn = _json.loads(raw_pn)
                    if not isinstance(raw_pn, list):
                        raw_pn = [int(raw_pn)]
                except (ValueError, _json.JSONDecodeError):
                    raw_pn = [int(x) for x in raw_pn.split(",") if x.strip().isdigit()]
            elif isinstance(raw_pn, (int, float)):
                raw_pn = [int(raw_pn)]

            image_minio_url = meta.get("image_minio_url", "") or None
            sources.append({
                "text": doc.page_content[:500],
                "filename": meta.get("filename", "Unknown"),
                "page_numbers": raw_pn,
                "heading": meta.get("heading"),
                "minio_url": meta.get("minio_url"),
                "image_url": image_minio_url,
            })

        return {
            "answer": answer,
            "sources": sources,
            "used_rag": used_rag,
        }

    async def analyze_and_suggest(self, conversation_id: str) -> str:
        """Generate quick suggestions for what can be done with uploaded files."""
        try:
            files = self.db_service.get_conversation_files(conversation_id)
            ready_files = [f for f in files if f.summary]
            if not ready_files:
                return "Files are still being processed. Please wait a moment and try again."

            from langchain_core.prompts import ChatPromptTemplate

            summary_text = "\n".join(
                [f"- **{f.filename}**: {f.summary}" for f in ready_files[:5]]
            )
            prompt = ChatPromptTemplate.from_template(
                "Based on these uploaded documents, what analyses, comparisons, "
                "or insights would be most valuable?\n\nDocuments:\n{summaries}\n\n"
                "Provide 3-4 specific, actionable suggestions:"
            )
            chain = prompt | self.llm
            response = await chain.ainvoke({"summaries": summary_text})
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return "Unable to generate suggestions at this time."


# Singleton
_agent_rag_service: Optional[AgentRAGService] = None


def get_agent_rag_service() -> AgentRAGService:
    """Get or create agent-based RAG service instance."""
    global _agent_rag_service
    if _agent_rag_service is None:
        _agent_rag_service = AgentRAGService()
    return _agent_rag_service
