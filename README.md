# Automated Counterstatement Generation against Misinformation via Generative AI

## Requirements
- **Python 3.11**
- Paket manager: **uv**
- **Ollama** running in the background

## Model Configuration



### Remote Model
**Mistral**
1) Set Up API key: You need to add an environment variable for your API key. (named MISTRAL_API_KEY)

Ubuntu example:

```bash
export MISTRAL_API_KEY="your_api_key_here"
```

### Local Model
**For now, we use for our RAG pipeline prototype a local Ollama LLM (here: *Llama3*). This might change during the project.**
1) Download Ollama: https://ollama.com/download.
2) Follow the on-screen instructions. Ollama installs and runs in the background. Once finished, you will see a tray icon for it in your taskbar.
3) Open a terminal on your computer.
4) Run the following command to download the latest version of **Llama 3**: ``ollama pull llama3``.
5) If you want to use the model in the terminal, run the following command to call the **Llama 3** model: ``ollama run llama3``.
Otherwise, just run the RAG pipeline. Ollama must run for model inference. Else you will get an error.


## Adding Data
To add data to the data base you can use the script *adding_data.py* using the command
``python3 adding_data.py data.csv --persist-dir chroma_db``
where ``data.csv`` contains the data that you want to add and chroma_db is the existing database.

If no vector database exists the script creates a new one. 

In the current database we used a chunk size of 1000 and overlap of 200. The encoder model we used is sentence-transformers/all-mpnet-base-v2 from Huggingface.


## Attention
If the VRAM of your GPU is not sufficient (e.g. 2 GB), Llama 3 will not work because it is 4.7 GB big.<br>
Instead, you could try a smaller model, e.g. **TinyLlama**.<br>
Run this ``ollama pull tinyllama`` and you should be good to go!