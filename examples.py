"""
Example usage of the LangChain LLM QA Bot
Developed by Abdessamad Bourkibate | IBM Certified AI Engineer

This script demonstrates different ways to use the PDF QA Bot.
"""

import os
from qa_bot import PDFQABot


def example_basic_usage():
    """
    Basic usage example: Load a PDF and ask questions.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize the bot
    bot = PDFQABot()
    
    # Path to your PDF file
    pdf_path = "sample_document.pdf"
    
    if os.path.exists(pdf_path):
        # Process the PDF
        bot.process_pdf_and_setup(pdf_path, persist=True)
        
        # Ask a question
        question = "What is this document about?"
        result = bot.ask(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"\nNumber of source documents: {len(result['source_documents'])}")
    else:
        print(f"Sample PDF not found: {pdf_path}")


def example_multiple_questions():
    """
    Example: Ask multiple questions to the same document.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multiple Questions")
    print("=" * 60)
    
    bot = PDFQABot()
    pdf_path = "sample_document.pdf"
    
    if os.path.exists(pdf_path):
        bot.process_pdf_and_setup(pdf_path)
        
        questions = [
            "What are the main objectives?",
            "What methodology was used?",
            "What are the key findings?",
            "What are the recommendations?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}] {question}")
            result = bot.ask(question)
            print(f"[Answer] {result['answer']}")
    else:
        print(f"Sample PDF not found: {pdf_path}")


def example_load_existing_vectorstore():
    """
    Example: Load an existing vector store instead of recreating it.
    """
    print("\n" + "=" * 60)
    print("Example 3: Using Existing Vector Store")
    print("=" * 60)
    
    bot = PDFQABot()
    
    try:
        # Load existing vector store
        bot.load_vectorstore()
        bot.setup_qa_chain()
        
        # Ask questions
        question = "Can you provide a summary?"
        result = bot.ask(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run example_basic_usage() first to create a vector store.")


def example_with_source_inspection():
    """
    Example: Inspect the source documents used to generate the answer.
    """
    print("\n" + "=" * 60)
    print("Example 4: Inspecting Source Documents")
    print("=" * 60)
    
    bot = PDFQABot()
    pdf_path = "sample_document.pdf"
    
    if os.path.exists(pdf_path):
        bot.process_pdf_and_setup(pdf_path)
        
        question = "What are the conclusions?"
        result = bot.ask(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"\nSource Documents:")
        print("-" * 60)
        
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n[Source {i}]")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    else:
        print(f"Sample PDF not found: {pdf_path}")


def example_custom_configuration():
    """
    Example: Using custom configuration for the bot.
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Initialize bot with custom settings
    bot = PDFQABot(
        model_name="gpt-3.5-turbo",
        embedding_model="text-embedding-ada-002",
        chroma_db_dir="./custom_chroma_db"
    )
    
    pdf_path = "sample_document.pdf"
    
    if os.path.exists(pdf_path):
        bot.process_pdf_and_setup(pdf_path, persist=True)
        
        question = "What is the key takeaway?"
        result = bot.ask(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
    else:
        print(f"Sample PDF not found: {pdf_path}")


def interactive_mode():
    """
    Interactive mode: Ask questions in a loop.
    """
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    print("\nInitializing QA Bot...")
    bot = PDFQABot()
    bot.process_pdf_and_setup(pdf_path)
    
    print("\nQA Bot is ready! Ask questions (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            result = bot.ask(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"(Based on {len(result['source_documents'])} source documents)")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("LangChain LLM QA Bot - Example Usage")
    print("Developed by Abdessamad Bourkibate | IBM Certified AI Engineer")
    print()
    
    # Choose which example to run
    print("Available examples:")
    print("1. Basic Usage")
    print("2. Multiple Questions")
    print("3. Load Existing Vector Store")
    print("4. Inspect Source Documents")
    print("5. Custom Configuration")
    print("6. Interactive Mode")
    print()
    
    choice = input("Select an example (1-6): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_multiple_questions()
    elif choice == "3":
        example_load_existing_vectorstore()
    elif choice == "4":
        example_with_source_inspection()
    elif choice == "5":
        example_custom_configuration()
    elif choice == "6":
        interactive_mode()
    else:
        print("Invalid choice. Please run again and select 1-6.")
