# Architecture Overview

## LangChain LLM QA Bot Architecture

This document provides a detailed overview of the architecture and design decisions behind the LangChain LLM QA Bot.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  (examples.py, qa_bot.py main(), interactive mode)              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer (PDFQABot)                  │
│  - Document Loading & Processing                                 │
│  - Vector Store Management                                       │
│  - QA Chain Configuration                                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangChain Framework                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Document  │  │   Text     │  │  Retrieval │               │
│  │  Loaders   │  │  Splitter  │  │    QA      │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   OpenAI API Services    │    │   ChromaDB Vector Store  │
│  - GPT Models (LLM)      │    │  - Embeddings Storage    │
│  - text-embedding-ada-002│    │  - Similarity Search     │
└──────────────────────────┘    └──────────────────────────┘
```

## Core Components

### 1. PDFQABot Class

The main orchestrator class that manages the entire QA pipeline.

**Responsibilities:**
- Configuration management (API keys, model selection)
- Document loading and preprocessing
- Vector store creation and management
- QA chain setup and execution
- Result aggregation and formatting

**Key Methods:**
- `load_pdf()`: Loads and chunks PDF documents
- `create_vectorstore()`: Creates embeddings and stores in ChromaDB
- `setup_qa_chain()`: Configures the retrieval QA pipeline
- `ask()`: Processes questions and returns answers with sources

### 2. Document Processing Pipeline

**Step 1: PDF Loading**
- Uses LangChain's `PyPDFLoader`
- Extracts text from each page
- Preserves metadata (page numbers, source)

**Step 2: Text Chunking**
- Uses `RecursiveCharacterTextSplitter`
- Chunk size: 1000 characters
- Overlap: 200 characters
- Maintains context between chunks

**Step 3: Embedding Generation**
- Converts text chunks to vector embeddings
- Uses OpenAI's `text-embedding-ada-002` model
- Each chunk becomes a dense vector representation

### 3. Vector Store (ChromaDB)

**Features:**
- Persistent storage option
- Fast similarity search using vector embeddings
- Metadata filtering capabilities
- Efficient retrieval of top-k relevant documents

**Storage Modes:**
- **Persistent**: Saved to disk, can be reused across sessions
- **In-Memory**: Temporary, faster for single-use cases

### 4. Retrieval-Augmented Generation (RAG)

**RAG Pipeline Steps:**

1. **Query Embedding**: User question is converted to a vector
2. **Similarity Search**: ChromaDB finds most similar document chunks (default k=4)
3. **Context Assembly**: Retrieved chunks are combined with the question
4. **LLM Generation**: GPT model generates answer based on provided context
5. **Source Attribution**: Original chunks are returned for transparency

**Benefits:**
- Answers grounded in actual document content
- Reduces hallucination
- Provides source references
- Scalable to large document collections

### 5. LangChain Integration

**Components Used:**

- **Document Loaders**: `PyPDFLoader` for PDF processing
- **Text Splitters**: `RecursiveCharacterTextSplitter` for chunking
- **Embeddings**: `OpenAIEmbeddings` for vector generation
- **Vector Stores**: `Chroma` for storage and retrieval
- **LLMs**: `ChatOpenAI` for answer generation
- **Chains**: `RetrievalQA` for end-to-end QA pipeline
- **Prompts**: `PromptTemplate` for custom prompt engineering

## Data Flow

### Document Indexing Flow

```
PDF File → PyPDFLoader → Raw Documents → RecursiveCharacterTextSplitter → 
Text Chunks → OpenAIEmbeddings → Vector Embeddings → ChromaDB
```

### Question Answering Flow

```
User Question → OpenAIEmbeddings → Query Vector → ChromaDB Similarity Search →
Top-k Relevant Chunks → PromptTemplate → ChatOpenAI → Answer + Sources
```

## Configuration Management

### Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `OPENAI_MODEL`: LLM model selection (default: gpt-3.5-turbo)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-ada-002)
- `CHROMA_DB_DIR`: Vector store directory (default: ./chroma_db)

### Constructor Parameters

All environment variables can be overridden via constructor parameters:

```python
bot = PDFQABot(
    openai_api_key="sk-...",
    model_name="gpt-4",
    embedding_model="text-embedding-ada-002",
    chroma_db_dir="./custom_db"
)
```

## Performance Considerations

### Optimization Strategies

1. **Vector Store Persistence**
   - Reuse previously created embeddings
   - Avoid re-processing documents
   - Faster startup times

2. **Chunk Size Tuning**
   - Larger chunks: More context, slower retrieval
   - Smaller chunks: More precise, faster retrieval
   - Current: 1000 chars with 200 overlap (balanced)

3. **Retrieval Parameters**
   - `k` parameter: Number of chunks to retrieve
   - Default k=4 provides good balance
   - Increase for more comprehensive answers
   - Decrease for faster response times

### Resource Usage

**Memory:**
- Embeddings: ~1KB per chunk
- Model: ~100MB for OpenAI API client
- ChromaDB: Minimal in-memory overhead

**API Costs:**
- Embedding: ~$0.0001 per 1K tokens
- GPT-3.5-turbo: ~$0.002 per 1K tokens
- Cost scales with document size and query volume

## Security Considerations

1. **API Key Management**
   - Never commit `.env` file to version control
   - Use environment variables or secure vaults
   - Rotate keys regularly

2. **Document Privacy**
   - Documents are sent to OpenAI for embedding/generation
   - Consider data sensitivity before uploading
   - Use private deployments for sensitive data

3. **Input Validation**
   - File existence checks before processing
   - Error handling for malformed PDFs
   - API key validation on initialization

## Extensibility

### Adding New Document Types

```python
# Add support for Word documents
from langchain_community.document_loaders import Docx2txtLoader

def load_docx(self, docx_path: str) -> List:
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()
    # ... rest of processing
```

### Custom Embedding Models

```python
# Use a different embedding provider
from langchain_community.embeddings import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Alternative Vector Stores

```python
# Use FAISS instead of ChromaDB
from langchain_community.vectorstores import FAISS

self.vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=self.embeddings
)
```

## Error Handling

The implementation includes comprehensive error handling:

- **File Not Found**: Clear error messages for missing files
- **API Key Issues**: Validation and helpful error messages
- **Vector Store Issues**: Proper checks before operations
- **Chain Not Setup**: Guards against using uninitialized components

## Testing Recommendations

### Unit Tests

- Test PDF loading with various file formats
- Test chunking with different configurations
- Test vector store creation and retrieval
- Test QA chain with mock embeddings

### Integration Tests

- End-to-end test with sample PDFs
- Test vector store persistence
- Test multi-session usage
- Test error conditions

### Performance Tests

- Benchmark embedding generation speed
- Benchmark retrieval latency
- Test with large documents (100+ pages)
- Memory usage profiling

## Future Enhancements

Potential improvements for the system:

1. **Multi-Document Support**: Query across multiple PDFs simultaneously
2. **Streaming Responses**: Stream answers as they're generated
3. **Caching**: Cache frequent queries for faster responses
4. **Batch Processing**: Process multiple questions efficiently
5. **Advanced Retrieval**: Implement hybrid search (keyword + semantic)
6. **UI/UX**: Build a web interface or CLI tool
7. **Monitoring**: Add logging and analytics
8. **Fine-tuning**: Support for custom fine-tuned models

## Conclusion

This architecture provides a robust, scalable foundation for building intelligent QA systems. The modular design allows for easy customization and extension while maintaining simplicity for end users.
