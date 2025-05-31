# 🕸️ AI Crawler with LangChain & Ollama

A web application that crawls web pages and enables AI-powered Q&A about their content using LangChain and Ollama.

## 🌟 Features

- Web page crawling and indexing
- AI-powered question answering
- Interactive chat interface
- Configurable text chunking
- Context-aware responses
- Local LLM integration

## 📋 Prerequisites

- Python 3.8+
- Ollama with llama2 model
- Chrome/Chromium browser
- macOS/Windows/Linux

## 🚀 Installation

### For macOS

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Pull the llama2 model
ollama pull llama2

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### For Windows

```batch
# Download and install Ollama from https://ollama.ai

# Start Ollama service from Windows Terminal
ollama serve

# Pull the llama2 model
ollama pull llama2

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `requirements.txt` file:

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-ollama
selenium
unstructured
validators
```

## 🎯 Usage

1. Start Ollama service:
```bash
ollama serve
```

2. Launch the application:
```bash
streamlit run ai_scraper.py
```

3. Open your browser to `http://localhost:8501`

## 💻 Application Flow

1. Enter a URL in the input field
2. Click "Crawl and Index Page"
3. Configure settings in sidebar:
   - Chunk Size (500-2000)
   - Chunk Overlap (50-500)
   - Retrieved Documents (1-10)
4. Ask questions in the chat interface
5. View AI responses with source context

## ⚡ Quick Settings

- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Default retrieved docs: 3
- Model: llama2

## 🔧 Troubleshooting

### Ollama Issues
- Verify Ollama is running: `ollama list`
- Check model installation: `ollama pull llama2`
- Restart Ollama service if needed

### Python Environment
- Remove existing venv: `rm -rf venv`
- Create fresh venv: `python3 -m venv venv`
- Reinstall dependencies: `pip install -r requirements.txt`

### Selenium Issues
- Update Chrome browser
- Verify ChromeDriver compatibility
- Check internet connection

## 📁 Project Structure

```
AI_Scraper/
├── ai_scraper.py      # Main application
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📝 License

MIT License

## 🔗 Links

- [Ollama](https://ollama.ai)
- [LangChain](https://python.langchain.com)
- [Streamlit](https://streamlit.io)
