import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA, ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client

class LangChainRAGSystem:
    def __init__(self):
        """Initialize the RAG System with Milvus, LangChain, Gemini, and LangSmith."""
        
        # Load environment variables
        load_dotenv()
        
        # Environment configurations
        self.milvus_host = os.getenv("MILVUS_HOST")
        self.milvus_port = os.getenv("MILVUS_PORT")
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv("LANGSMITH_PROJECT_NAME")
        self.emb_model=os.getenv("EMBEDDING_MODEL")

        # Initialize components
        self._setup_langsmith()
        self._setup_embeddings()
        self._setup_milvus()
        self._setup_llm()
        self._setup_qa_chain()
        self._setup_chat_memory()

    def _setup_langsmith(self):
        """Initialize LangSmith tracking if API key is available."""
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            self.tracer = LangChainTracer(project_name=self.project_name)
            self.langsmith_client = Client()
            print("LangSmith tracking enabled!")
        else:
            self.tracer = None
            self.langsmith_client = None
            print("LangSmith tracking not configured.")

    def _setup_embeddings(self):
        """Initialize Gemini embeddings."""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

    def _setup_milvus(self):
        """Initialize Milvus vector store connection."""
        print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
        self.milvus = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "host": self.milvus_host,
                "port": self.milvus_port
            }
        )
        self.retriever = self.milvus.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print(f"Connected to Milvus collection: {self.collection_name}")

    def _setup_llm(self):
        """Initialize Gemini model."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=self.google_api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            max_output_tokens=2048,
        )

    def _setup_chat_memory(self):
        """Initialize chat memory and conversation chain."""
        template = """You are a helpful AI assistant with access to a knowledge base. 
        Use the retrieved information to answer questions accurately.

        Current conversation:
        {history}
        Human: {input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )

        self.memory = ConversationBufferMemory(
            return_messages=True,
            ai_prefix="Assistant",
            human_prefix="Human"
        )

        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )

    def _setup_qa_chain(self):
            """Initialize the RAG chain with a custom prompt that includes context."""
            rag_prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say you don't know.

            Context:
            {context}

            Question: {question}

            Answer:"""
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=rag_prompt_template,
                        input_variables=["context", "question"]
                    ),
                }
            )


    def query(self, question: str, use_memory: bool = True):
        """
        Query the RAG system with a user question.
        
        Args:
            question (str): User's question
            use_memory (bool): Whether to use conversation memory
            
        Returns:
            dict: Response containing answer, source documents, and similarity scores
        """
        try:
            # 1. Retrieve relevant documents from Milvus
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found in the knowledge base.",
                    "source_documents": [],
                    "similarity_scores": []
                }

            # 2. Format the retrieved documents into context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 3. Get similarity scores for the retrieved documents
            similarity_scores = self.milvus.similarity_search_with_score(
                question,
                k=3
            )

            # 4. Generate response using the LLM
            if use_memory:
                # Add context to conversation memory
                context_prompt = f"Context: {context}\nQuestion: {question}"
                response = self.conversation_chain.predict(input=context_prompt)
                
                return {
                    "answer": response,
                    "source_documents": relevant_docs,
                    "similarity_scores": similarity_scores
                }
            else:
                # Use the QA chain with explicit context
                response = self.qa_chain({
                    "query": question,
                    "context": context
                })
                
                return {
                    "answer": response["result"],
                    "source_documents": relevant_docs,
                    "similarity_scores": similarity_scores
                }

        except Exception as e:
            print(f"Error during RAG query: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_documents": [],
                "similarity_scores": []
            }

    def similarity_search(self, query: str, k: int = 3):
        """
        Perform a similarity search in Milvus without generating an LLM response.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[tuple]: List of (document, similarity_score) tuples
        """
        try:
            results = self.milvus.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []


    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared")

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = LangChainRAGSystem()
    
    # # Example: Full RAG query
    # question = "What are the admission requirements for the MS in Applied Data Science program?"
    # response = rag_system.query(question=question)
    # print("Answer:", response["answer"])
    # print("\nSource Documents:")
    # for doc, score in zip(response["source_documents"], response["similarity_scores"]):
    #     print(f"- Document: {doc.page_content}")
    #     print(f"  Similarity Score: {score}")
    #     print(f"  Metadata: {doc.metadata}")
    
    # Example: Similarity search only
    print("\nSimilarity Search Results:")
    search_results = rag_system.similarity_search("capstone project")
    for doc, score in search_results:
        print(f"- Document: {doc.page_content}")
        print(f"  Similarity Score: {score}")