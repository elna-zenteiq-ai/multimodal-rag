"""Vision captioning service using NVIDIA VLM endpoint."""

import base64
import logging

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import get_settings

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this image in detail. Include all visible text, data values, "
    "labels, axis names, legends, and structural elements. "
    "If it is a chart or diagram, explain what it represents. "
    "Be factual and thorough — your description will be used for search."
)


class VisionService:
    """Generate text captions for images using an NVIDIA VLM."""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatNVIDIA(
            model=settings.nvidia_vision_model,
            api_key=settings.nvidia_api_key,
            temperature=0.3,
            max_completion_tokens=1024,
        )

    async def caption_image(
        self, image_bytes: bytes, context: str = ""
    ) -> str:
        """Generate a text caption for an image.

        Args:
            image_bytes: Raw PNG/JPEG bytes.
            context: Optional surrounding text context from the document.

        Returns:
            A descriptive text caption.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompt_text = _CAPTION_PROMPT
        if context:
            prompt_text += f"\n\nSurrounding document context: {context}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        try:
            response = await self.llm.ainvoke([message])
            caption = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Generated caption ({len(caption)} chars)")
            return caption
        except Exception as e:
            logger.error(f"Vision captioning failed: {e}")
            # Fallback: return the context or a placeholder
            if context:
                return f"[Image] {context}"
            return "[Image — caption unavailable]"


# Singleton
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    """Get or create the VisionService singleton."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
