# DocuRAG Pipeline
# PDF → Documents → Chunks → Embeddings → Vector DB → Retriever → LLM → Answer

import os
import numpy as np
from pathlib import Path

# ============================================================
# STEP 1: Document Loading (PDF/TXT → Documents)
# ============================================================
def load_document(file_path: str) -> str:
    """Load document from PDF or text file."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        # Use PyPDF2 for PDF loading
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except ImportError:
            return "Install PyPDF2: pip install PyPDF2"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ============================================================
# STEP 2: Text Chunking (Documents → Chunks)
# ============================================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

# ============================================================
# STEP 3: Embeddings Generation (Chunks → Embeddings)
# ============================================================
def generate_embeddings(chunks: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for text chunks using sentence transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings)
    except ImportError:
        return np.zeros((len(chunks), 384))  # Fallback for demo

# ============================================================
# STEP 4: Vector Database (Embeddings → Vector DB)
# ============================================================
class SimpleVectorDB:
    """Simple in-memory vector database for demonstration."""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.metadata = []
    
    def add(self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict] = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = metadata or [{}] * len(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Search for most similar chunks using cosine similarity."""
        if self.embeddings is None:
            return []
        
        # Cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.chunks[i], similarities[i]) for i in top_indices]

# ============================================================
# STEP 5: Retriever (Vector DB → Context)
# ============================================================
def retrieve_context(query: str, vector_db: SimpleVectorDB, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[str]:
    """Retrieve relevant context from vector database."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        query_embedding = model.encode([query])
        
        results = vector_db.search(query_embedding[0], top_k=3)
        return [chunk for chunk, score in results]
    except ImportError:
        return []

# ============================================================
# STEP 6: LLM + Context → Answer
# ============================================================def extract_answer_from_context(query: str, context: str) -> str:
    """Simple keyword-based answer extraction when LLM fails."""
    query_lower = query.lower()
    context_lines = context.replace('. ', '.\n').split('\n')
    
    # Keywords to look for based on query
    if "leave" in query_lower:
        keywords = ["leave", "casual", "sick", "earned"]
    elif "working hours" in query_lower or "work hours" in query_lower:
        keywords = ["working hours", "9:00", "6:00", "login"]
    elif "salary" in query_lower or "pay" in query_lower:
        keywords = ["salary", "credited", "month", "pf", "insurance"]
    elif "work from home" in query_lower or "wfh" in query_lower:
        keywords = ["work from home", "home", "manager approval"]
    elif "notice" in query_lower or "exit" in query_lower:
        keywords = ["notice period", "exit", "settlement"]
    elif "it policy" in query_lower or "password" in query_lower:
        keywords = ["it policy", "password", "company systems"]
    else:
        keywords = []
    
    # Find relevant sentences
    relevant = []
    for line in context_lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            relevant.append(line.strip())
    
    if relevant:
        return " | ".join(relevant)
    else:
        # Return first 200 chars as summary
        return context[:200] + "..."
def generate_answer(query: str, context_chunks: list[str], llm_provider: str = "gemini") -> str:
    """Generate answer using LLM with retrieved context."""
    context = "\n\n".join(context_chunks)
    
    if llm_provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "Install openai: pip install openai"
    
    elif llm_provider == "gemini":
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "Set GEMINI_API_KEY in your environment before using the Gemini provider."
            
            genai.configure(api_key=api_key)
            
            # Try different model names - gemini-2.0-flash is the latest
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            prompt = f"""Based on the following context, answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}\n\nContext retrieved:\n{context}"
    
    elif llm_provider == "huggingface":
        try:
            from huggingface_hub import InferenceClient
            
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return "Set HF_TOKEN in your environment before using the Hugging Face provider."
            
            client = InferenceClient(token=hf_token)
            
            prompt = f"""Based on the following context, answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            # Use a model that's available on free inference API
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: Extract relevant info from context using simple keyword matching
            return extract_answer_from_context(query, context)
    
    elif llm_provider == "ollama":
        try:
            import requests
            
            # Connect to local Ollama server
            url = "http://localhost:11434/api/generate"
            
            prompt = f"""Based on the following context, answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            payload = {
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500}
            }
            
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Ollama Error: {response.status_code}\n\nContext:\n{context}"
        except Exception as e:
            return f"Ollama Error: {str(e)}\n\nMake sure Ollama is installed and running.\n\nContext:\n{context}"
    
    # Fallback: Simple rule-based response
    return f"""Based on retrieved context:

{context}

---
Answer to "{query}": Please configure an LLM provider API key in your environment for generation."""

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_docurag_pipeline(document_path: str, query: str, llm_provider: str = "huggingface"):
    """Run complete DocuRAG pipeline."""
    print("=" * 60)
    print("DocuRAG Pipeline")
    print("=" * 60)
    
    # Step 1: Load Document
    print("\n[1/6] Loading document...")
    text = load_document(document_path)
    print(f"   Loaded {len(text)} characters")
    
    # Step 2: Chunk Text
    print("\n[2/6] Chunking text...")
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Generate Embeddings
    print("\n[3/6] Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Step 4: Store in Vector DB
    print("\n[4/6] Storing in vector database...")
    vector_db = SimpleVectorDB()
    vector_db.add(chunks, embeddings)
    print(f"   Stored {len(chunks)} vectors")
    
    # Step 5: Retrieve Context
    print("\n[5/6] Retrieving relevant context...")
    context_chunks = retrieve_context(query, vector_db)
    print(f"   Retrieved {len(context_chunks)} context chunks")
    
    # Step 6: Generate Answer
    print("\n[6/6] Generating answer...")
    answer = generate_answer(query, context_chunks, llm_provider)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    return answer

# Run the pipeline
if __name__ == "__main__":
    # Example usage
    document_path = "policy.txt"
    query = "What are the working hours? and When is salary credited?"
    
    answer = run_docurag_pipeline(document_path, query)
    print(answer)
