from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

from document_retriever import retrieve_docs

def format_context(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def build_llm(model: str):
    print("Loading local model...")
    llm = ChatOllama(model=model)
    prompt = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Answer by using the provided context."
            "If you are unsure of the answer, just say that you don't know and don't make up an answer."),
        HumanMessagePromptTemplate.from_template("Question: {question}\n\nContext:\n{context}"),
    ])
    chain = prompt | llm
    return chain

def generate_answer(
    *,
    question: str,
    retriever,
    chain,
):
    docs = retrieve_docs(retriever, question)
    context = format_context(docs)
    print("Generating answer...")
    answer = (chain.invoke({"question": question, "context": context})).content
    return answer

# there are many more models provided by langchain
# like e.g. llama2, llama3, gpt-4, etc.