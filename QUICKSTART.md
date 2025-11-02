# Quick Start Guide

Get started with LangChain LLM QA Bot in 5 minutes!

## Step 1: Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/ABDESSAMAD-BOURKIBATE/LangChain-LLM-QA-Bot.git
cd LangChain-LLM-QA-Bot

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure (1 minute)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

## Step 3: Run Your First Query (2 minutes)

### Option A: Use the Interactive Mode

```bash
python examples.py
# Select option 6 (Interactive Mode)
# Enter the path to your PDF file
# Start asking questions!
```

### Option B: Use Python Code

Create a file named `my_qa.py`:

```python
from qa_bot import PDFQABot

# Initialize the bot
bot = PDFQABot()

# Process your PDF
bot.process_pdf_and_setup("your_document.pdf")

# Ask a question
result = bot.ask("What is this document about?")
print(result['answer'])
```

Run it:
```bash
python my_qa.py
```

## Sample Questions to Try

- "What is the main topic of this document?"
- "Can you summarize the key findings?"
- "What methodology was used?"
- "What are the conclusions?"
- "Who are the authors?"
- "What are the limitations mentioned?"

## Common Use Cases

### Research Papers
```python
bot.process_pdf_and_setup("research_paper.pdf")
result = bot.ask("What are the main contributions of this paper?")
```

### Technical Documentation
```python
bot.process_pdf_and_setup("api_documentation.pdf")
result = bot.ask("How do I authenticate API requests?")
```

### Books and Textbooks
```python
bot.process_pdf_and_setup("textbook.pdf")
result = bot.ask("Explain the concept of neural networks.")
```

## Tips for Best Results

1. **Ask Specific Questions**: More specific questions get better answers
   - ❌ "What does it say?"
   - ✅ "What are the three main benefits discussed in section 2?"

2. **Check Sources**: Always review the source documents
   ```python
   result = bot.ask("Your question")
   print(f"Answer: {result['answer']}")
   print(f"Based on {len(result['source_documents'])} sources")
   ```

3. **Reuse Vector Stores**: Process documents once, query many times
   ```python
   # First time - create and persist
   bot.process_pdf_and_setup("doc.pdf", persist=True)
   
   # Later - just load and query
   bot.load_vectorstore()
   bot.setup_qa_chain()
   bot.ask("Your question")
   ```

## Troubleshooting

### "OpenAI API key not provided"
→ Make sure your `.env` file has `OPENAI_API_KEY=your_key`

### "PDF file not found"
→ Use absolute path or ensure file is in current directory
```python
import os
pdf_path = os.path.abspath("your_file.pdf")
bot.process_pdf_and_setup(pdf_path)
```

### Slow first query
→ This is normal - the bot is creating embeddings. Subsequent queries are faster.

### Out of memory
→ For very large PDFs, consider processing in batches or increasing chunk size

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
- Run [examples.py](examples.py) to see more usage patterns
- Explore advanced features like custom models and chain types

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check README.md and ARCHITECTURE.md first
- **API Limits**: Monitor your OpenAI usage at https://platform.openai.com/usage

---

**That's it! You're ready to build intelligent QA applications.** 🚀

*Developed by Abdessamad Bourkibate | IBM Certified AI Engineer*
