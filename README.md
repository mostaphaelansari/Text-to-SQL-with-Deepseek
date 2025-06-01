# Text-to-SQL Application

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Gradio](https://img.shields.io/badge/gradio-FF6B35?style=for-the-badge&logo=gradio&logoColor=white)

A web-based application that converts natural language questions into SQL queries and executes them against SQLite databases.

## Features

- Natural language to SQL query conversion
- SQLite database connection and schema analysis
- Query execution with result display
- Conversation history tracking
- Sample database creation for testing
- Direct SQL query execution interface

## Requirements

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

**Core Dependencies:**
- ![SQLite](https://img.shields.io/badge/sqlite3-built--in-green)
- ![Pandas](https://img.shields.io/badge/pandas-latest-blue)
- ![LangChain](https://img.shields.io/badge/langchain--core-latest-orange)
- ![LangChain Ollama](https://img.shields.io/badge/langchain--ollama-latest-orange)
- ![Gradio](https://img.shields.io/badge/gradio-latest-red)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install pandas langchain-core langchain-ollama gradio
```
3. Ensure Ollama is installed and running with the deepseek-coder model:
```bash
ollama pull deepseek-coder:6.7b
```

## Usage

### Starting the Application

```bash
python app.py
```

The application will launch in your web browser at `http://localhost:7860`

### Database Connection

1. Upload a SQLite database file (.db, .sqlite, .sqlite3)
2. Or create a sample e-commerce database for testing

### Querying Data

Submit questions in natural language through the Query Interface:
- "Show all customers from USA"
- "List top 5 products by sales"
- "Find orders over $500"

### Direct SQL Execution

Use the SQL Executor tab to run custom SQL queries directly against the connected database.

## Configuration

The application uses the `deepseek-coder:6.7b` model by default. To change the model:

```python
app = TextToSQLApp(model_name="your-preferred-model")
```

## Sample Database Schema

The application can generate a sample e-commerce database with the following tables:
- `customers` - Customer information
- `products` - Product catalog
- `orders` - Order records
- `order_items` - Order line items

## Limitations

- Supports SQLite databases only
- Requires Ollama runtime with compatible language models
- Query complexity depends on model capabilities
- No authentication or user management

## Architecture

![Architecture](https://img.shields.io/badge/Architecture-Component--Based-lightgrey)

- **Backend**: ![Python](https://img.shields.io/badge/Python-FFD43B?logo=python&logoColor=blue) with ![LangChain](https://img.shields.io/badge/LangChain-121212?logo=langchain) for LLM integration
- **Database**: ![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?logo=sqlite&logoColor=white) with ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white) for data handling
- **Frontend**: ![Gradio](https://img.shields.io/badge/Gradio-FF6B35?logo=gradio&logoColor=white) web interface
- **LLM**: ![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white) compatible models for SQL generation

## Error Handling

The application includes error handling for:
- Database connection failures
- Invalid SQL syntax
- Missing database schema
- Model communication errors
