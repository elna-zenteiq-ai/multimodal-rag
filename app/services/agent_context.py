from dataclasses import dataclass


@dataclass
class AgentContext:
    """Context schema provided to agents and tools at runtime.

    - `user_id` can be used to scope personalization or permissions
    - `conversation_id` is used to scope searches and tools to a conversation
    """
    user_id: str | None = None
    conversation_id: str | None = None
