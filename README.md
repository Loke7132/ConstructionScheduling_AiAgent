# AI Construction Scheduling Agent

A sophisticated AI-powered agent that analyzes construction project documents and automatically generates detailed project schedules. The agent uses advanced language models, retrieval-augmented generation (RAG), and reasoning frameworks to extract project requirements from PDF documents and create comprehensive construction schedules with tasks, dependencies, durations, and resource requirements.

## 🚀 Features

- **Intelligent Document Analysis**: Processes PDF construction documents using OCR and text extraction
- **AI-Powered Schedule Generation**: Uses Cerebras API with Llama-3.3-70b model for intelligent reasoning
- **Retrieval-Augmented Generation (RAG)**: Semantic search through document content using vector embeddings
- **ReAct Framework**: Follows Thought-Action-Observation pattern for systematic reasoning
- **Automatic API Key Rotation**: Handles rate limits with intelligent key rotation system
- **Comprehensive Task Management**: Identifies tasks, dependencies, durations, and resource requirements
- **Robust Error Handling**: Graceful handling of API failures, document processing errors, and edge cases
- **JSON Export**: Outputs detailed schedules in structured JSON format

## 📋 Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- Valid Cerebras API key(s)
- PDF construction documents to analyze

### System Dependencies

**For macOS:**
```bash
brew install tesseract
```

**For Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**For Windows:**
- Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

## 🛠️ Installation

1. **Clone or download the project files:**
   ```bash
   git clone <repository-url>
   cd CAgent
   ```

2. **Install required Python packages:**
   ```bash
   pip install langchain chromadb sentence-transformers PyMuPDF pytesseract requests python-dotenv
   ```

   Or using the full dependency list:
   ```bash
   pip install langchain==0.1.0 chromadb==0.4.0 sentence-transformers==2.2.0 PyMuPDF==1.23.0 pytesseract==0.3.10 requests==2.31.0 python-dotenv==1.0.0
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   # Primary Cerebras API key (required)
   CEREBRAS_API_KEY_1=your_primary_api_key_here
   
   # Additional API keys for rotation (optional but recommended)
   CEREBRAS_API_KEY_2=your_backup_api_key_here
   CEREBRAS_API_KEY_3=your_third_api_key_here
   ```

## ⚙️ Configuration

Before running the agent, ensure you have:

1. **API Keys**: At least one valid Cerebras API key set in environment variables
2. **PDF Documents**: Construction project documents in PDF format
3. **Document Path**: Update the `pdf_path` variable in the `main()` function

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CEREBRAS_API_KEY_1` | Yes | Primary Cerebras API key |
| `CEREBRAS_API_KEY_2` | No | Backup API key for rotation |
| `CEREBRAS_API_KEY_3` | No | Additional backup API key |

## 🚀 Usage

### Basic Usage

1. **Prepare your PDF document:**
   - Ensure your construction project document is in PDF format
   - Place it in an accessible directory

2. **Update the script configuration:**
   ```python
   # In the main() function, update this line:
   pdf_path = "path/to/your/construction_document.pdf"
   ```

3. **Run the agent:**
   ```bash
   python construction_agent.py
   ```

### Expected Output

The agent will:
1. Process the PDF document and extract text
2. Create a vector database for semantic search
3. Initialize the AI agent with reasoning capabilities
4. Analyze the document and generate a construction schedule
5. Export the final schedule as a JSON file

### Sample Output Structure

```json
{
  "project_info": {
    "name": "Office Building Construction",
    "description": "Commercial office building project",
    "start_date": "2024-01-15",
    "estimated_duration": "18 months"
  },
  "tasks": [
    {
      "id": "TASK_001",
      "name": "Site Preparation",
      "description": "Clear and level construction site",
      "duration_days": 10,
      "dependencies": [],
      "resources": ["excavator", "bulldozer", "workers"]
    }
  ],
  "dependencies": [
    {
      "task_id": "TASK_002",
      "depends_on": ["TASK_001"],
      "dependency_type": "finish_to_start"
    }
  ]
}
```

## 🏗️ Architecture Overview

### Core Components

1. **CerebrasAPIKeyRotator**: Manages multiple API keys and handles rate limiting
2. **DocumentProcessor**: Extracts text from PDFs and creates vector embeddings
3. **ScheduleState**: Maintains agent's working memory for schedule building
4. **Llama70BWithRotation**: Custom LangChain LLM wrapper with API rotation
5. **Agent Tools**: Specialized functions for document search and schedule management

### AI Agent Tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Semantic search through document content |
| `add_task` | Add new tasks to the schedule |
| `add_dependency` | Define task dependencies |
| `update_task_duration` | Modify task duration estimates |
| `add_resource_requirement` | Specify required resources |
| `get_current_schedule` | View current schedule state |
| `set_project_info` | Set project metadata |
| `finish_schedule` | Complete and export the schedule |

### ReAct Framework Flow

```
Thought: Analyze what needs to be done
Action: Choose and execute appropriate tool
Observation: Review results and plan next step
[Repeat until schedule is complete]
```

## 🔧 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: No valid Cerebras API keys found
```
- Ensure `CEREBRAS_API_KEY_1` is set in your environment
- Verify the API key is valid and has sufficient credits

**2. PDF Processing Errors**
```
Error: Could not process PDF document
```
- Verify the PDF file exists and is readable
- Ensure Tesseract OCR is properly installed
- Check if the PDF contains extractable text

**3. ChromaDB Initialization Issues**
```
Error: Could not initialize vector store
```
- Ensure sufficient disk space for ChromaDB storage
- Check write permissions in the project directory

**4. Tool Parsing Errors**
```
Could not parse LLM output
```
- This is handled automatically with `handle_parsing_errors=True`
- The agent will retry with corrected formatting

### Performance Optimization

- **Multiple API Keys**: Use 2-3 API keys for better rate limit handling
- **Document Size**: For large PDFs, consider splitting into smaller sections
- **Memory Usage**: Monitor memory usage with very large documents

## 📊 Supported Document Types

The agent works best with:
- Construction project specifications
- Architectural drawings with text annotations
- Project requirement documents
- Contractor proposals and bids
- Technical specifications
- Project timelines and milestones

## 🔐 Security Considerations

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` files for local development
- **Document Privacy**: Ensure sensitive documents comply with your organization's policies
- **API Usage**: Monitor API usage to avoid unexpected charges

## 📈 Advanced Usage

### Custom Tool Development

You can extend the agent by adding custom tools:

```python
@tool
def custom_analysis_tool(query: str) -> str:
    """Custom tool for specialized analysis"""
    # Your custom logic here
    return result
```

### Integration with Project Management Systems

The JSON output can be integrated with:
- Microsoft Project
- Primavera P6
- Asana/Trello
- Custom project management systems

## 🤝 Contributing

To contribute to this project:
1. Follow PEP 8 coding standards
2. Add comprehensive docstrings
3. Include error handling for new features
4. Test with various PDF document types

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error logs for specific error messages
3. Ensure all dependencies are properly installed
4. Verify API key configuration

## 🔄 Version History

- **v1.0.0**: Initial release with core RAG and ReAct functionality
- **v1.1.0**: Added automatic API key rotation
- **v1.2.0**: Enhanced PDF processing with OCR fallback
- **v1.3.0**: Improved error handling and tool reliability

---

**Built with:** Python, LangChain, ChromaDB, Cerebras API, PyMuPDF, and Tesseract OCR 