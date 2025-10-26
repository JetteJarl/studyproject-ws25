from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

def build_llm(model: str):
    llm = ChatOllama(model=model)
    prompt = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant. Answer using provided context if relevant."),
        HumanMessagePromptTemplate.from_template("Question: {question}\n\nContext:\n{context}"),
    ])
    chain = prompt | llm
    return chain

# there are many more models provided by langchain
# like e.g. llama2, llama3, gpt-4, etc.