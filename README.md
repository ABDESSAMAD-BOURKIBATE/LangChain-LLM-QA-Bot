# LangChain-LLM-QA-Bot

**Developed by Abdessamad Bourkibate | IBM Certified AI Engineer**

An intelligent Question-Answering (QA) Bot that leverages Retrieval-Augmented Generation (RAG) techniques with LangChain and Large Language Models (LLMs) to answer questions from uploaded research documents using vector embeddings and ChromaDB.

## 🌟 Features

- **PDF Document Processing**: Automatically extracts and processes text from PDF files
- **Vector Embeddings**: Converts document chunks into vector embeddings using OpenAI's embedding models
- **ChromaDB Integration**: Stores and retrieves embeddings efficiently using ChromaDB vector database
- **RAG Pipeline**: Implements Retrieval-Augmented Generation for context-aware answers
- **LangChain Framework**: Built on top of LangChain for robust LLM orchestration
- **Source Attribution**: Returns source documents used to generate answers
- **Flexible Configuration**: Supports custom models, embedding strategies, and storage options

## 🏗️ Architecture

The QA Bot follows a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Loading**: PDFs are loaded and parsed using PyPDFLoader
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Question Processing**: User questions are embedded and used to retrieve relevant chunks
6. **Answer Generation**: Retrieved context is passed to the LLM to generate accurate answers

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   PDF File  │─────▶│ Text Chunks  │─────▶│ Embeddings  │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Answer    │◀─────│     LLM      │◀─────│  ChromaDB   │
└─────────────┘      └──────────────┘      └─────────────┘
                            ▲
                            │
                     ┌──────────────┐
                     │   Question   │
                     └──────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ABDESSAMAD-BOURKIBATE/LangChain-LLM-QA-Bot.git
   cd LangChain-LLM-QA-Bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 💻 Usage

### Basic Usage

```python
from qa_bot import PDFQABot

# Initialize the bot
bot = PDFQABot()

# Process a PDF file
bot.process_pdf_and_setup("your_document.pdf")

# Ask questions
result = bot.ask("What is the main topic of this document?")
print(result['answer'])
```

### Advanced Usage

```python
from qa_bot import PDFQABot

# Initialize with custom configuration
bot = PDFQABot(
    model_name="gpt-4",
    embedding_model="text-embedding-ada-002",
    chroma_db_dir="./my_vectorstore"
)

# Load and process PDF
documents = bot.load_pdf("research_paper.pdf")
print(f"Loaded {len(documents)} chunks")

# Create vector store
bot.create_vectorstore(documents, persist=True)

# Set up QA chain
bot.setup_qa_chain()

# Ask questions and inspect sources
result = bot.ask("What methodology was used in this research?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['source_documents'])} documents")

# View source documents
for i, doc in enumerate(result['source_documents']):
    print(f"\nSource {i+1}:")
    print(doc.page_content[:200])
```

### Using Existing Vector Store

```python
from qa_bot import PDFQABot

# Initialize bot
bot = PDFQABot()

# Load existing vector store (instead of recreating)
bot.load_vectorstore()
bot.setup_qa_chain()

# Ask questions
result = bot.ask("Can you summarize the key findings?")
print(result['answer'])
```

### Interactive Mode

Run the examples file for an interactive experience:

```bash
python examples.py
```

Select option 6 for interactive mode where you can ask questions in real-time.

## 📁 Project Structure

```
LangChain-LLM-QA-Bot/
│
├── qa_bot.py              # Main QA Bot implementation
├── examples.py            # Usage examples and interactive mode
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
├── LICENSE               # License file
└── README.md             # This file
```

## 🔧 Configuration

You can configure the bot using environment variables or constructor parameters:

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
CHROMA_DB_DIR=./chroma_db
```

### Constructor Parameters

```python
bot = PDFQABot(
    openai_api_key="your_key",      # OpenAI API key
    model_name="gpt-4",              # LLM model to use
    embedding_model="text-embedding-ada-002",  # Embedding model
    chroma_db_dir="./vectorstore"    # Vector store directory
)
```

## 📚 Key Components

### PDFQABot Class

The main class that orchestrates the QA system:

- **`load_pdf(pdf_path)`**: Loads and chunks a PDF document
- **`create_vectorstore(documents, persist=True)`**: Creates a vector store from documents
- **`load_vectorstore()`**: Loads an existing vector store from disk
- **`setup_qa_chain(chain_type="stuff")`**: Sets up the QA retrieval chain
- **`ask(question)`**: Asks a question and returns an answer with sources
- **`process_pdf_and_setup(pdf_path, persist=True)`**: Convenience method to do everything in one call

## 🎯 Use Cases

- **Research Paper Analysis**: Extract insights from academic papers
- **Document Summarization**: Get quick summaries of lengthy documents
- **Legal Document Review**: Ask questions about contracts and legal documents
- **Technical Documentation**: Query technical manuals and guides
- **Educational Materials**: Study from textbooks and course materials

## 🔍 How It Works

1. **Document Processing**:
   - PDF is loaded using PyPDFLoader
   - Text is split into chunks (1000 chars with 200 char overlap)
   - Each chunk maintains metadata (page number, source)

2. **Embedding Creation**:
   - Each text chunk is converted to a vector embedding
   - Uses OpenAI's text-embedding-ada-002 model
   - Embeddings capture semantic meaning of text

3. **Vector Storage**:
   - Embeddings stored in ChromaDB (persistent or in-memory)
   - Enables fast similarity search
   - Can be reused across sessions

4. **Question Answering**:
   - User question is embedded using the same model
   - Most similar document chunks are retrieved (top-k)
   - Retrieved chunks + question sent to LLM
   - LLM generates answer based on provided context

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'langchain'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `ValueError: OpenAI API key not provided`
- **Solution**: Set `OPENAI_API_KEY` in your `.env` file or pass it to the constructor

**Issue**: `FileNotFoundError: PDF file not found`
- **Solution**: Ensure the PDF path is correct and the file exists

**Issue**: ChromaDB persistence issues
- **Solution**: Ensure the directory specified in `CHROMA_DB_DIR` is writable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdessamad Bourkibate**
- IBM Certified AI Engineer
- GitHub: [@ABDESSAMAD-BOURKIBATE](https://github.com/ABDESSAMAD-BOURKIBATE)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [OpenAI](https://openai.com/) - LLM and embedding models
- [ChromaDB](https://www.trychroma.com/) - Vector database

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Note**: Remember to keep your OpenAI API key secure and never commit it to version control.
