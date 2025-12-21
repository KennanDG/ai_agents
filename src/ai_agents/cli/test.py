from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1",
    temperature=0.7
)

response = model.invoke("What is today's date")

print(response.content)