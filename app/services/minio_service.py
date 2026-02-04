"""MinIO service for raw file storage."""

import uuid
from datetime import timedelta
from io import BytesIO

from minio import Minio
from minio.error import S3Error

from app.config import get_settings


class MinioService:
    """Service for interacting with MinIO object storage."""

    def __init__(self):
        """Initialize MinIO client and ensure bucket exists."""
        settings = get_settings()
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            raise RuntimeError(f"Failed to create bucket: {e}") from e

    def upload_document(self, file_content: bytes, filename: str) -> tuple[str, str]:
        """Upload a document to MinIO.

        Args:
            file_content: Raw file bytes
            filename: Original filename

        Returns:
            Tuple of (object_name, presigned_url)
        """
        # Generate unique object name
        doc_id = str(uuid.uuid4())
        extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
        object_name = f"{doc_id}/{filename}"

        # Upload file
        try:
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=BytesIO(file_content),
                length=len(file_content),
                content_type=self._get_content_type(extension),
            )
        except S3Error as e:
            raise RuntimeError(f"Failed to upload file: {e}") from e

        # Generate presigned URL
        presigned_url = self.get_document_url(object_name)

        return doc_id, presigned_url

    def get_document_url(self, object_name: str, expires: int = 7) -> str:
        """Get a presigned URL for a document.

        Args:
            object_name: Object path in MinIO
            expires: URL expiration in days

        Returns:
            Presigned URL
        """
        try:
            return self.client.presigned_get_object(
                bucket_name=self.bucket,
                object_name=object_name,
                expires=timedelta(days=expires),
            )
        except S3Error as e:
            raise RuntimeError(f"Failed to generate presigned URL: {e}") from e

    def list_documents(self) -> list[dict]:
        """List all documents in the bucket.

        Returns:
            List of document info dicts
        """
        documents = []
        try:
            objects = self.client.list_objects(self.bucket, recursive=True)
            for obj in objects:
                # Extract document ID and filename from object name
                parts = obj.object_name.split("/", 1)
                if len(parts) == 2:
                    doc_id, filename = parts
                    documents.append({
                        "document_id": doc_id,
                        "filename": filename,
                        "minio_url": self.get_document_url(obj.object_name),
                        "uploaded_at": obj.last_modified.isoformat() if obj.last_modified else None,
                    })
        except S3Error as e:
            raise RuntimeError(f"Failed to list documents: {e}") from e

        return documents

    def check_health(self) -> bool:
        """Check if MinIO is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.bucket_exists(self.bucket)
            return True
        except Exception:
            return False

    @staticmethod
    def _get_content_type(extension: str) -> str:
        """Get MIME type for file extension."""
        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "txt": "text/plain",
            "md": "text/markdown",
            "html": "text/html",
        }
        return content_types.get(extension.lower(), "application/octet-stream")


# Singleton instance
_minio_service: MinioService | None = None


def get_minio_service() -> MinioService:
    """Get or create MinIO service instance."""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinioService()
    return _minio_service
