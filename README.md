# Automated Counterstatement Generation against Misinformation via Generative AI

## Requirements
- **Python 3.11**
- Paket manager: **uv**


## Setup and Run
1) **Clone the repository**
```bash
  git clone <your-repo-url>
  cd <your-repo-folder>
```

2) **Initialize UV and Install requirements**
```bash
  uv init
  uv install
```

3) **(Optional) Add Mistral API Key**

    When you are using the repository with the native LLMs we provide, you will need to setup a Mistral API key in your system. 
    If you prefer using another provider you can find instructions on how to add a new model below.

    ```bash
    export MISTRAL_API_KEY="your_api_key_here"
    ```

    OR

    Create an .env file (file must be really named ".env", dont forget the dot!) without file ending

    Enter this: MISTRAL_API_KEY="your_api_key_here"

    **IMPORTANT**: Never push the .env file or else the key might get leaked!

    Keep it as a local file.

4) **Run Streamlit Application**

    ```bash
    streamlit run rag_pipeline.py
    ```
    Open your browser to view the application. 


## Model Configuration


**Mistral**
1) Set Up API key: You need to add an environment variable for your API key. (named MISTRAL_API_KEY)

Ubuntu example:

```bash
export MISTRAL_API_KEY="your_api_key_here"
```

## Add Models (ADVANCED)
You can add models manually by changing the source code. 
To add a new model do the following 

1) **Model Class (ONLY for non Mistral models)**
   
   Add and implement a new llm class that inherits from ``__LlmModel__``.
   Implement the methods ``__build_llm()__`` and ``__generate_answer()__``.

  ``__build_llm()__`` needs to be called in the constructor (``__init()__``) and instantiates the llm.

  ``__generate_answer()__`` is called whenever the application querys the llm.
  
2) **Register Model**

     Register the llm by adding the model name and model class to the dictionary ``all_llms``



## Adding Data
To add data to the data base you can use the script *adding_data.py* using the command
``uv run python adding_data.py <path-to-data/data.csv>``
where ``data.csv`` contains the data that you want to add and chroma_db is the existing database.

If no vector database exists, the script creates a new one. Any data that already exists in the database will be skipped and not added again. To add new data, you can either create a new CSV file or append new rows to the existing one.
Make sure the CSV file contains a column named "url" with the links to the websites that should be ingested.

In the current database we used a chunk size of 1000 and overlap of 200. The encoder model we used is sentence-transformers/all-mpnet-base-v2 from Huggingface.
