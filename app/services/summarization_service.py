"""Service for file summarization using LLM."""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import get_settings

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service for summarizing documents using NVIDIA LLM."""

    SUMMARIZATION_PROMPT = """You are an expert document analyst. Summarize the following document content in a clear, concise manner.

Focus on:
1. Main topics and themes
2. Key information and insights
3. Important conclusions or findings

Keep the summary to 3-5 sentences maximum, capturing the essential information.

Document Content:
{content}

Summary:"""

    def __init__(self):
        """Initialize summarization service with NVIDIA LLM."""
        self.settings = get_settings()
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=0.3,  # Lower temp for consistent summaries
            top_p=0.9,
            max_tokens=500,
        )
        self.prompt = ChatPromptTemplate.from_template(self.SUMMARIZATION_PROMPT)

    def summarize(self, content: str, max_tokens: int = 500) -> Optional[str]:
        """Summarize document content.

        Args:
            content: Document text content
            max_tokens: Maximum tokens for summary

        Returns:
            Summary string or None if failed
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for summarization")
            return None

        try:
            # Truncate content if too long (to avoid token limit)
            max_input_chars = 4000  # Roughly 1000 tokens
            if len(content) > max_input_chars:
                content = content[: max_input_chars] + "..."

            chain = self.prompt | self.llm
            response = chain.invoke({"content": content})
            summary = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Generated summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return None

    async def asummarize(self, content: str, max_tokens: int = 500) -> Optional[str]:
        """Async version of summarize.

        Args:
            content: Document text content
            max_tokens: Maximum tokens for summary

        Returns:
            Summary string or None if failed
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for summarization")
            return None

        try:
            # Truncate content if too long
            max_input_chars = 4000
            if len(content) > max_input_chars:
                content = content[: max_input_chars] + "..."

            chain = self.prompt | self.llm
            response = await chain.ainvoke({"content": content})
            summary = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Generated summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Error during async summarization: {e}")
            return None


# Singleton instance
_summarization_service: SummarizationService | None = None


def get_summarization_service() -> SummarizationService:
    """Get or create summarization service instance."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service
