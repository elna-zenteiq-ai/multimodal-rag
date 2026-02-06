"""Agent-based RAG service using LangChain agent framework."""

import logging
from typing import Optional

from langchain import agents
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.database import MessageIntentType, get_db_service
from app.services.intent_classifier import get_intent_classifier
from app.services.milvus_service import get_milvus_service
from app.services.summary_search import get_summary_search_service

logger = logging.getLogger(__name__)


class AgentRAGService:
    """Agent-based RAG service with intent-driven routing and tool execution."""

    SYSTEM_PROMPT = """You are an intelligent document analysis assistant powered by retrieval and reasoning.

Your capabilities:
1. Answer questions based on provided document content and summaries
2. Perform detailed searches when deeper information is needed
3. Compare and analyze information across documents
4. Provide concise, accurate responses with proper citations

When a user provides a query:
- First, try to answer from the provided summaries if they contain sufficient information
- For factual questions requiring details, use the RAG search tool
- Always cite your sources when referencing specific information
- Be clear about limitations if information isn't available

Remember:
- Keep responses concise and focused
- Provide specific page numbers and file names when citing
- If multiple interpretations exist, ask for clarification
- Use the search tool to find detailed information when needed"""

    def __init__(self):
        """Initialize agent-based RAG service."""
        self.settings = get_settings()
        self.milvus = get_milvus_service()
        self.db_service = get_db_service()
        self.intent_classifier = get_intent_classifier()
        self.summary_search = get_summary_search_service()

        # Initialize LLM with NVIDIA
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=self.settings.nvidia_temperature,
            top_p=self.settings.nvidia_top_p,
            max_tokens=self.settings.nvidia_max_tokens,
        )

    def _create_rag_tool(self, file_ids: list[str]) -> tool:
        """Create RAG search tool scoped to specific files.

        Args:
            file_ids: List of file IDs to search within

        Returns:
            LangChain Tool object
        """
        def rag_search_scoped(query: str, top_k: int = 5) -> str:
            """Search documents scoped to specific files."""
            try:
                results = self.milvus.search(query=query, top_k=top_k)

                # Filter to scoped files
                if file_ids:
                    results = [d for d in results if d.metadata.get("file_id") in file_ids]

                if not results:
                    return "No relevant information found in the specified documents."

                # Format results
                from app.services.rag_tool import _format_search_results
                return _format_search_results(results, file_ids)

            except Exception as e:
                logger.error(f"Error in RAG search: {e}")
                return f"Error searching documents: {str(e)}"

        return tool(
            name="rag_search",
            func=rag_search_scoped,
            description="Search through documents to find relevant information. Use this for detailed questions that need specific facts from the documents.",
        )

    async def query_with_files(
        self,
        query: str,
        conversation_id: str,
        file_ids: Optional[list[str]] = None,
        chat_history: Optional[list] = None,
    ) -> dict:
        """Query the agent with optional file context.

        Args:
            query: User query
            conversation_id: Conversation ID
            file_ids: List of file IDs available in context (if any)
            chat_history: Previous conversation messages

        Returns:
            Dict with keys: answer, sources, intent, used_rag
        """
        if not chat_history:
            chat_history = []

        # Get conversation files if not specified
        if file_ids is None:
            files = self.db_service.get_conversation_files(conversation_id)
            file_ids = [f.file_id for f in files if f.summary]

        logger.info(f"Processing query: {query[:100]}... with {len(file_ids)} files")

        # 1. Classify intent
        classification = await self.intent_classifier.aclassify(query)
        intent = classification["intent"]
        is_ambiguous = classification["is_ambiguous"]

        if is_ambiguous:
            return {
                "answer": f"I need clarification: {classification['clarification_question']}",
                "sources": [],
                "intent": MessageIntentType.AMBIGUOUS,
                "used_rag": False,
            }

        # 2. Route based on intent
        if intent == MessageIntentType.HIGH_LEVEL:
            # For high-level questions, use summaries
            answer = await self._answer_from_summaries(query, conversation_id, file_ids, chat_history)
            return {
                "answer": answer,
                "sources": [],
                "intent": intent,
                "used_rag": False,
            }

        else:  # FACTUAL
            # For factual questions, use agent with RAG tool
            answer, used_rag = await self._answer_with_agent(
                query, conversation_id, file_ids, chat_history
            )
            return {
                "answer": answer,
                "sources": [],
                "intent": intent,
                "used_rag": used_rag,
            }

    async def _answer_from_summaries(
        self, query: str, conversation_id: str, file_ids: list[str], chat_history: list
    ) -> str:
        """Answer question using file summaries without RAG search.

        Args:
            query: User query
            conversation_id: Conversation ID
            file_ids: Available file IDs
            chat_history: Previous messages

        Returns:
            Answer string
        """
        try:
            # Get summaries for the files
            summaries = self.summary_search.get_conversation_summaries(conversation_id)

            if not summaries:
                return "I don't have any documents with summaries to answer from. Please try uploading documents first."

            # Format summaries as context
            summary_context = "\n\n".join(
                [f"**File: {file_id}**\n{summary}" for file_id, summary in summaries]
            )

            # Create simple prompt for summary-based answering
            prompt = ChatPromptTemplate.from_template(
                """Based on these document summaries, answer the following question. 
                
Document Summaries:
{summaries}

Question: {query}

Answer:"""
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "summaries": summary_context,
                "query": query,
            })

            answer = response.content if hasattr(response, "content") else str(response)
            return answer

        except Exception as e:
            logger.error(f"Error answering from summaries: {e}")
            return f"Error processing query: {str(e)}"

    async def _answer_with_agent(
        self, query: str, conversation_id: str, file_ids: list[str], chat_history: list
    ) -> tuple[str, bool]:
        """Answer question using agent with RAG tool access.

        Args:
            query: User query
            conversation_id: Conversation ID
            file_ids: Available file IDs
            chat_history: Previous messages

        Returns:
            Tuple of (answer, used_rag_tool)
        """
        try:
            # Create RAG tool scoped to available files
            tools = [self._create_rag_tool(file_ids)]

            # Create agent using new LangChain v1 API
            agent = agents.create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=self.SYSTEM_PROMPT,
            )

            # Run agent (new API returns direct response, no executor needed)
            result = await agent.ainvoke({
                "input": query,
                "chat_history": chat_history,
            })

            answer = result.get("output", "Unable to generate answer")
            # Check if RAG tool was used in the agent execution
            used_rag = "rag_search" in str(result)

            return answer, used_rag

        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            return f"Error processing query: {str(e)}", False

    async def analyze_and_suggest(self, conversation_id: str) -> str:
        """Analyze uploaded files and suggest possible analyses.

        Args:
            conversation_id: Conversation ID

        Returns:
            Suggestions text
        """
        try:
            files = self.db_service.get_conversation_files(conversation_id)
            ready_files = [f for f in files if f.summary]

            if not ready_files:
                return "Files are still being processed. Please wait a moment and try again."

            # Create summary text
            summary_text = "\n".join(
                [f"- **{f.filename}**: {f.summary}" for f in ready_files[:5]]
            )

            prompt = ChatPromptTemplate.from_template(
                """Based on these uploaded documents, what analyses, comparisons, or insights would be most valuable?

Documents:
{summaries}

Provide 3-4 specific, actionable suggestions for what analyses could be performed:"""
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({"summaries": summary_text})

            suggestions = response.content if hasattr(response, "content") else str(response)
            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return "Unable to generate suggestions at this time."


# Singleton instance
_agent_rag_service: Optional[AgentRAGService] = None


def get_agent_rag_service() -> AgentRAGService:
    """Get or create agent-based RAG service instance."""
    global _agent_rag_service
    if _agent_rag_service is None:
        _agent_rag_service = AgentRAGService()
    return _agent_rag_service
