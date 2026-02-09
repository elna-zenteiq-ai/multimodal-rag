"""Service for intent classification and message analysis."""

import logging
import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.database import MessageIntentType

logger = logging.getLogger(__name__)


class IntentClassifierService:
    """Service for classifying message intent and detecting ambiguity."""

    CLASSIFICATION_PROMPT = """You are a message intent classifier. Analyze the user message and classify its intent.

Respond in JSON format with exactly these fields:
{{
    "intent": "HIGH_LEVEL" or "FACTUAL",
    "is_ambiguous": true or false,
    "clarification_question": "question text or null"
}}

Intent Types:
- HIGH_LEVEL: Requests for overview, summary, comparison, general insights,
  or any broad question that can be answered from document summaries.
  This includes short commands like "summarize", "overview", "what is this about?",
  "give me a summary", "key points", "main findings", etc.
- FACTUAL: Specific factual questions that require detailed retrieval and search
  through document chunks (e.g. asking about a particular number, date, name, quote,
  section, or specific detail).

Ambiguity: Set to true ONLY if the message is genuinely unclear and you cannot
determine what the user wants. Short imperative commands like "summarize" are
NOT ambiguous — they clearly request a summary of the available documents.

Examples:
- "summarize" → HIGH_LEVEL, is_ambiguous=false
- "Summarize the main findings" → HIGH_LEVEL, is_ambiguous=false
- "Give me an overview" → HIGH_LEVEL, is_ambiguous=false
- "What is this document about?" → HIGH_LEVEL, is_ambiguous=false
- "Key takeaways" → HIGH_LEVEL, is_ambiguous=false
- "Compare document A and B" → HIGH_LEVEL, is_ambiguous=false
- "What are the differences between document A and B?" → HIGH_LEVEL, is_ambiguous=false
- "What specific data point X was mentioned?" → FACTUAL, is_ambiguous=false
- "What was the revenue in Q3?" → FACTUAL, is_ambiguous=false
- "What does section 4.2 say?" → FACTUAL, is_ambiguous=false
- "Tell me about it" → AMBIGUOUS (unclear which document/aspect)

User Message:
{message}

Classification:"""

    def __init__(self):
        """Initialize intent classifier with NVIDIA LLM."""
        from app.config import get_settings
        self.settings = get_settings()
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=0.3,  # Low temp for consistent classification
            top_p=0.9,
            max_tokens=200,
        )
        self.prompt = ChatPromptTemplate.from_template(self.CLASSIFICATION_PROMPT)

    def classify(self, message: str) -> dict:
        """Classify message intent and ambiguity.

        Args:
            message: User message to classify

        Returns:
            Dict with keys: intent (MessageIntentType), is_ambiguous (bool), clarification_question (str|None)
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"message": message})
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            result = self._parse_response(response_text)
            return result

        except Exception as e:
            logger.error(f"Error during intent classification: {e}")
            # Default to FACTUAL if error
            return {
                "intent": MessageIntentType.FACTUAL,
                "is_ambiguous": False,
                "clarification_question": None,
            }

    async def aclassify(self, message: str) -> dict:
        """Async version of classify.

        Args:
            message: User message to classify

        Returns:
            Dict with keys: intent (MessageIntentType), is_ambiguous (bool), clarification_question (str|None)
        """
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"message": message})
            response_text = response.content if hasattr(response, "content") else str(response)

            result = self._parse_response(response_text)
            return result

        except Exception as e:
            logger.error(f"Error during async intent classification: {e}")
            return {
                "intent": MessageIntentType.FACTUAL,
                "is_ambiguous": False,
                "clarification_question": None,
            }

    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response to extract JSON classification.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed classification dict
        """
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Map intent string to enum
                intent_str = data.get("intent", "FACTUAL").upper()
                if intent_str == "HIGH_LEVEL":
                    intent = MessageIntentType.HIGH_LEVEL
                elif intent_str == "FACTUAL":
                    intent = MessageIntentType.FACTUAL
                else:
                    intent = MessageIntentType.FACTUAL

                return {
                    "intent": intent,
                    "is_ambiguous": data.get("is_ambiguous", False),
                    "clarification_question": data.get("clarification_question"),
                }
        except Exception as e:
            logger.warning(f"Failed to parse classification response: {e}")

        # Default response
        return {
            "intent": MessageIntentType.FACTUAL,
            "is_ambiguous": False,
            "clarification_question": None,
        }


# Singleton instance
_intent_classifier: IntentClassifierService | None = None


def get_intent_classifier() -> IntentClassifierService:
    """Get or create intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifierService()
    return _intent_classifier
