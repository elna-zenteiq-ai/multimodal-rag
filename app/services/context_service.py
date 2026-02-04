class ContextService:
    def resolve_context(
        self,
        conversation_id: str,
        message_embedding,
        message_id: str,
    ) -> str | None:
        files = list_files_for_conversation(conversation_id)

        if len(files) == 0:
            return None

        if len(files) == 1:
            commit(conversation_id, files[0], message_id)
            return files[0]

        scores = compare_with_file_embeddings(message_embedding)

        if confident(scores):
            commit(conversation_id, top_file, message_id)
            return top_file

        return None  # clarification required
