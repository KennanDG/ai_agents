import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Configuration ---
FILE_PATH = "../../../data/about_me.md"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"


def main():
    # 1. Load the Document
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    loader = TextLoader(FILE_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")

    # 2. Split the Text (Chunking)
    # We split by 500 characters with some overlap to preserve context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # 3. Initialize Embeddings & Vector Store
    # This sends your chunks to Ollama to be converted into vectors
    print("Embedding documents... (this might take a moment)")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # We use Chroma (a local vector DB) to store the vectors in memory for this test
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )

    # Create a retriever that searches for the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Define the LLM and Prompt
    llm = ChatOllama(model=CHAT_MODEL)
    
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build the Chain (The "Pipeline")
    # flow: Question -> Retrieve Context -> Format Prompt -> LLM -> String Output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Run Queries
    print("\n--- RAG System Ready ---")
    questions = [
        "What is DeAngelo's dream school?",
        "Does he know Terraform?",
        "What is his current hardware setup?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        response = rag_chain.invoke(q)
        print(f"Answer: {response}")



if __name__ == "__main__":
    main()