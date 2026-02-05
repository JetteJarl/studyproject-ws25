# Automated Counterstatement Generation against Misinformation via Generative AI

## Requirements
- **Python 3.11**
- Paket manager: **uv**
- **Ollama** running in the background

## Model Configuration


**Mistral**
1) Set Up API key: You need to add an environment variable for your API key. (named MISTRAL_API_KEY)

Ubuntu example:

```bash
export MISTRAL_API_KEY="your_api_key_here"
```

## Adding Data
To add data to the data base you can use the script *adding_data.py* using the command
``uv run python adding_data.py <path-to-data/data.csv>``
where ``data.csv`` contains the data that you want to add and chroma_db is the existing database.

If no vector database exists, the script creates a new one. Any data that already exists in the database will be skipped and not added again. To add new data, you can either create a new CSV file or append new rows to the existing one.
Make sure the CSV file contains a column named "url" with the links to the websites that should be ingested.

In the current database we used a chunk size of 1000 and overlap of 200. The encoder model we used is sentence-transformers/all-mpnet-base-v2 from Huggingface.
