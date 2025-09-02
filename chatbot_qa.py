from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama

def create_qa_chain():
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma(
        persist_directory="chroma_db_2",
        embedding = embedding_model
    )
    
    retriever = vectordb.as_retriever()
    
    llm = ChatOllama(model="llama3", temperature=0)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    return qa_chain

def main():
    qa_chain = create_qa_chain()
    
    print("Chatbot is ready enter 'exit' to end the conversation")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break
        
        response = qa_chain.invoke({"question" : user_input})
        print(f"Agent: {response['answer']}")

if __name__ == "__main__":
    main()
