from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant. Answer the question as best you can."),
    HumanMessagePromptTemplate.from_template("{input}"),
])

llm = ChatOllama(model="llama3")

chain = prompt | llm
answer = chain.invoke({
    "input": "What is the capital of France?",
}).content
print(answer)