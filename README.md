# Automated Counterstatement Generation against Misinformation via Generative AI

## Requirements
- **Python 3.11**
- Paket manager: **uv**
- **Ollama** running in the background

## For now, we use for our RAG pipeline prototype a local Ollama LLM (here: *Llama3*). This might change during the project.
1) Download Ollama: https://ollama.com/download.
2) Follow the on-screen instructions. Ollama installs and runs in the background. Once finished, you will see a tray icon for it in your taskbar.
3) Open a terminal on your computer.
4) Run the following command to download the latest version of **Llama 3**: ``ollama pull llama3``.
5) If you want to use the model in the terminal, run the following command to call the **Llama 3** model: ``ollama run llama3``.
Otherwise, just run the RAG pipeline. Ollama must run for model inference. Else you will get an error.