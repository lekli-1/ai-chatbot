import io
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import PyPDF2

from sqlalchemy.orm import sessionmaker, Session
from db_setup import engine, Document

load_dotenv()

app = FastAPI(title="Local Cloud-Native RAG API")
client = AsyncOpenAI()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Automatically handles opening and closing database connections."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChatRequest(BaseModel):
    message: str


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Reads PDF bytes and extracts text page by page."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Splits a long text string into smaller overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def process_pdf_and_store(file_bytes: bytes, filename: str):
    """Extracts text, generates embeddings, and saves everything to the database."""
    try:
        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            print(f"No text found in {filename}.")
            return

        chunks = chunk_text(text)
        print(f"Processed {filename} into {len(chunks)} chunks.")

        # Open a distinct database session specifically for this background task
        db = SessionLocal()

        # Delete old chunks for this file so we don't create duplicates!
        db.query(Document).filter(Document.filename == filename).delete()
        db.commit()

        # Generate vectors and save to database
        for chunk in chunks:
            response = await client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )

            doc = Document(
                filename=filename,
                content=chunk,
                embedding=response.data[0].embedding
            )
            db.add(doc)

        db.commit()
        db.close()
        print(f"Successfully saved {filename} to database.")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


# API ENDPOINTS:
@app.post("/upload-pdf",responses={400 : {"description": "Invalid file type"}})
async def upload_pdf(background_tasks: BackgroundTasks, file: Annotated[UploadFile, File(...)]):
    """Receives a PDF and queues it for background processing."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = await file.read()
    background_tasks.add_task(process_pdf_and_store, file_bytes, file.filename)

    return {"message": f"{file.filename} is being processed."}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Annotated[Session, Depends(get_db)]):
    """Handles user questions by searching the database and asking OpenAI."""

    # Convert user's question to vector
    question_res = await client.embeddings.create(
        input=request.message,
        model="text-embedding-3-small"
    )
    question_vector = question_res.data[0].embedding

    # Search database for closest matches
    closest_docs = db.query(Document).order_by(
        Document.embedding.cosine_distance(question_vector)
    ).limit(3).all()

    # 3. Create context for OpenAI
    context_chunks = [doc.content for doc in closest_docs]
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""
    You are a helpful assistant. Use ONLY the provided context below to answer the user's question. 
    If the answer is not contained in the context, say "I don't have enough information."

    CONTEXT:
    {context_text}
    """

    # Get response from OpenAI
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
    )

    return {"reply": response.choices[0].message.content}


@app.get("/documents")
async def get_documents(db: Annotated[Session, Depends(get_db)]):
    """An endpoint to view all stored document chunks."""
    all_docs = db.query(Document.id, Document.filename, Document.content).all()
    return {
        "total_chunks": len(all_docs),
        "documents": [
            {"id": doc.id, "file": doc.filename, "content": doc.content}
            for doc in all_docs
        ]
    }


@app.get("/files")
async def get_uploaded_files(db: Annotated[Session, Depends(get_db)]):
    """An endpoint to view a list of unique uploaded files."""
    files = db.query(Document.filename).distinct().all()
    return {
        "uploaded_files": [f[0] for f in files if f[0]]
    }