import io
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from api import app, get_db, Document

# Test Client
client = TestClient(app)

# Mock Database setup
class MockSession:
    """A mock database session to simulate behavior without PostgreSQL."""

    def query(self, *args, **kwargs):
        class DummyQuery:
            def distinct(self): return self

            def order_by(self, *o_args, **o_kwargs): return self

            def limit(self, *l_args, **l_kwargs): return self

            def all(self):
                # Simulate returning unique files for the /files endpoint
                if args and args[0] is Document.filename:
                    return [('presentation_slide.pdf',), ('annual_report.pdf',)]

                # Simulate returning full Document entities for /documents and /chat context
                class MockDoc:
                    def __init__(self, doc_id, filename, content):
                        self.id = doc_id
                        self.filename = filename
                        self.content = content

                return [
                    MockDoc(1, 'presentation_slide.pdf', 'This is the first extracted chunk covering AI architecture.'),
                    MockDoc(2, 'annual_report.pdf', 'This chunk covers the cloud-native vector database setup.')
                ]

        return DummyQuery()


def override_get_mock_db():
        yield MockSession()


# Override the app's database injected dependency globally for these tests
app.dependency_overrides[get_db] = override_get_mock_db


# TESTS

def test_get_documents():
    """Tests that the database dependency override successfully feeds data to the endpoint."""
    response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 2
    assert data["documents"][0]["file"] == "presentation_slide.pdf"
    assert "AI architecture" in data["documents"][0]["content"]


def test_upload_invalid():
    """Tests application input validation (FastAPI HTTP Exceptions)."""
    # Create fake text file bytes
    file_content = b"This is plain text, not a PDF."
    files = {
        "file": ("test.txt", io.BytesIO(file_content), "text/plain")
    }

    response = client.post("/upload-pdf", files=files)

    # Asserts that the app caught the wrong extension and returned 400 Bad Request
    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are allowed"}


@patch("api.client.chat.completions.create", new_callable=AsyncMock)
@patch("api.client.embeddings.create", new_callable=AsyncMock)
def test_chat_endpoint_rag(mock_embeddings, mock_chat):
    """
    Simulates a full RAG (Retrieval-Augmented Generation) chat request:
    Intersects the OpenAI calls to use mock embeddings and mock chat completions.
    """

    # Mock the Vector Embedding AI Response
    mock_embeddings_response = MagicMock()
    mock_embeddings_response.data = [MagicMock(embedding=[0.05] * 1536)]
    mock_embeddings.return_value = mock_embeddings_response

    # Mock the Chatbot Text AI Response
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [
        MagicMock(message=MagicMock(content="Based on the context, the architecture is cloud-native."))
    ]
    mock_chat.return_value = mock_chat_response

    # Trigger the actual chat endpoint
    payload = {
        "message": "Can you summarize the architecture?",
        "history": [{"role": "user", "content": "Hello"}]
    }
    response = client.post("/chat", json=payload)

    # Verify HTTP and JSON format behavior
    assert response.status_code == 200
    assert response.json() == {"reply": "Based on the context, the architecture is cloud-native."}

    # Verify that our endpoint correctly attempted to talk to the AI provider
    mock_embeddings.assert_called_once()
    mock_chat.assert_called_once()