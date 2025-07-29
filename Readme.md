# McDonald's Menu RAG System using Gemini and FAISS

A Retrieval-Augmented Generation (RAG) system for querying McDonald's menu data using Google's Gemini AI and FAISS vector database.

## Project Overview

This system processes McDonald's menu CSV files, creates embeddings using Gemini, and provides an interactive query interface to ask questions about menu items, categories, and pricing.

## Setup

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv myenv
.\myenv\Scripts\Activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory with your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

You can get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 4. Data Preparation

The system expects CSV files with the following required columns:
- `Menu Item` - Name of the menu item
- `Category` - Category of the menu item  
- `Price` - Price of the menu item

Place your McDonald's CSV files in the appropriate year folders under `data/`:
- `data/2022_filtered_files/`
- `data/2023_filtered_files/`
- `data/2024_filtered_files/`
- `data/2025_filtered_files/`

## Usage

### Step 1: Process and Chunk Data

Run the chunking script to process CSV files and create text chunks:

```bash
python scripts/chunk_menu.py
```

This script:
- Scans all CSV files in the data directory
- Groups items by category
- Creates JSONL chunks for embedding
- Outputs to `data/chunks.jsonl`

### Step 2: Create Embeddings and Index

Generate embeddings and build the FAISS index:

```bash
python scripts/embed_and_index.py
```

This script:
- Loads chunks from `data/chunks.jsonl`
- Creates embeddings using Gemini's embedding model
- Builds a FAISS index for similarity search
- Saves index to `data/menu_index.faiss`
- Saves metadata to `data/chunks_meta.json`

### Step 3: Query the RAG System

Start the interactive query interface:

```bash
python scripts/query_rag.py
```

Ask questions like:
- "What breakfast items are available?"
- "What's the price of Big Mac?"
- "Show me all chicken sandwiches"
- "What vegetarian options do you have?"

## Project Structure

```
McD_RAG/
├── data/                          # Data directory
│   ├── 2022_filtered_files/      # 2022 CSV files
│   ├── 2023_filtered_files/      # 2023 CSV files
│   ├── 2024_filtered_files/      # 2024 CSV files
│   ├── 2025_filtered_files/      # 2025 CSV files
│   ├── chunks.jsonl              # Processed text chunks
│   ├── menu_index.faiss          # FAISS vector index
│   └── chunks_meta.json          # Chunk metadata
├── scripts/
│   ├── chunk_menu.py             # Data processing script
│   ├── embed_and_index.py        # Embedding and indexing script
│   └── query_rag.py              # Interactive query interface
├── app/                          # Web application (if applicable)
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Dependencies

- `pandas` - Data processing
- `faiss-cpu` - Vector similarity search
- `google-generativeai` - Gemini AI API
- `python-dotenv` - Environment variable management
- `streamlit` - Web interface (optional)

## Troubleshooting

### Missing Required Columns
If you see "required columns missing" errors, ensure your CSV files contain:
- `Menu Item`
- `Category` 
- `Price`

### API Key Issues
- Verify your `.env` file contains `GEMINI_API_KEY`
- Check that your API key is valid and has sufficient quota

### Memory Issues
- For large datasets, consider processing files in smaller batches
- The system uses FAISS CPU version for compatibility

## Features

- **Multi-year data support** - Process data from 2022-2025
- **Category-based chunking** - Groups items by menu categories
- **Semantic search** - Find relevant menu items using natural language
- **Interactive interface** - Ask questions in plain English
- **Metadata tracking** - Track source files for each chunk