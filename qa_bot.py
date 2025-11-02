"""
LangChain LLM QA Bot
Developed by Abdessamad Bourkibate | IBM Certified AI Engineer

An intelligent Question-Answering Bot that leverages Retrieval-Augmented Generation (RAG)
techniques with LangChain and Large Language Models (LLMs) to answer questions from 
uploaded research documents using vector embeddings and ChromaDB.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class PDFQABot:
    """
    A Question-Answering Bot that uses RAG to extract answers from PDF documents.
    
    This bot processes PDF documents, creates vector embeddings, stores them in ChromaDB,
    and uses a retrieval-augmented generation approach to answer questions based on the
    document content.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        chroma_db_dir: str = "./chroma_db"
    ):
        """
        Initialize the PDF QA Bot.
        
        Args:
            openai_api_key: OpenAI API key (if not provided, will use from environment)
            model_name: Name of the OpenAI model to use for answering questions
            embedding_model: Name of the OpenAI embedding model
            chroma_db_dir: Directory to store ChromaDB vector store
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self.model_name = model_name if model_name is not None else os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.embedding_model = embedding_model if embedding_model is not None else os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.chroma_db_dir = chroma_db_dir if chroma_db_dir is not None else os.getenv("CHROMA_DB_DIR", "./chroma_db")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.embedding_model
        )
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def load_pdf(self, pdf_path: str) -> List:
        """
        Load and process a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed document chunks
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    def create_vectorstore(self, documents: List, persist: bool = True):
        """
        Create a vector store from documents using ChromaDB.
        
        Args:
            documents: List of document chunks
            persist: Whether to persist the vector store to disk
        """
        if persist:
            # Create persistent vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_dir
            )
        else:
            # Create in-memory vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
    
    def load_vectorstore(self):
        """
        Load an existing vector store from disk.
        """
        if not os.path.exists(self.chroma_db_dir):
            raise FileNotFoundError(
                f"Vector store not found at {self.chroma_db_dir}. "
                "Please create one first using create_vectorstore()."
            )
        
        self.vectorstore = Chroma(
            persist_directory=self.chroma_db_dir,
            embedding_function=self.embeddings
        )
    
    def setup_qa_chain(self, chain_type: str = "stuff"):
        """
        Set up the question-answering chain with retrieval.
        
        Args:
            chain_type: Type of chain to use ("stuff", "map_reduce", "refine", "map_rerank")
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store not initialized. Load or create a vector store first."
            )
        
        # Create custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer based on the loaded documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError(
                "QA chain not initialized. Call setup_qa_chain() first."
            )
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def process_pdf_and_setup(self, pdf_path: str, persist: bool = True):
        """
        Convenience method to process a PDF and set up the QA system in one call.
        
        Args:
            pdf_path: Path to the PDF file
            persist: Whether to persist the vector store to disk
        """
        print(f"Loading PDF: {pdf_path}")
        documents = self.load_pdf(pdf_path)
        print(f"Loaded {len(documents)} document chunks")
        
        print("Creating vector store...")
        self.create_vectorstore(documents, persist=persist)
        print("Vector store created")
        
        print("Setting up QA chain...")
        self.setup_qa_chain()
        print("QA Bot ready!")


def main():
    """
    Example usage of the PDF QA Bot.
    """
    # Initialize the bot
    bot = PDFQABot()
    
    # Example: Process a PDF file
    pdf_path = "your_document.pdf"
    
    if os.path.exists(pdf_path):
        # Process PDF and set up the bot
        bot.process_pdf_and_setup(pdf_path)
        
        # Ask questions
        questions = [
            "What is the main topic of this document?",
            "Can you summarize the key findings?",
            "What are the conclusions?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            result = bot.ask(question)
            print(f"A: {result['answer']}")
            print(f"Sources: {len(result['source_documents'])} documents")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path to use the QA Bot.")


if __name__ == "__main__":
    main()
