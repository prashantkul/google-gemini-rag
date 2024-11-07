import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import traceback
import requests
# Core LangChain imports
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# LangChain specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_openai import OpenAI 

# Memory and callbacks
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer

# LangSmith
from langsmith import Client

# Chain types for RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import google.generativeai as genai

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LangChainRAGSystem:
    def __init__(self):
        """Initialize the RAG System with Milvus, LangChain, Gemini, and LangSmith."""
        
        # Load environment variables
        load_dotenv()
        
        # Environment configurations
        self.milvus_host = os.getenv("MILVUS_HOST")
        self.milvus_port = os.getenv("MILVUS_PORT")
        self.milvus_db = os.getenv("MILVUS_DB")
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv("LANGSMITH_PROJECT_NAME")
        self.emb_model = os.getenv("EMBEDDING_MODEL")

        # Initialize components
        self._setup_langsmith()
        self._setup_embeddings()
        self._setup_milvus()
       # self._setup_llm()
        self._setup_retrieval_chain()
        self._setup_chat_memory()
        self.configure_gemini()
        self._setup_secondary_llm()


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


    def configure_gemini(self):
        """Configure the Gemini API with the provided API key."""
        if not self.google_api_key:
            raise ValueError("API Key is missing. Please provide a valid API key.")
        genai.configure(api_key=self.google_api_key)
        print("Gemini API configured successfully.")
    
    def _setup_secondary_llm(self):
        """Initialize ChatGPT-4 as the secondary LLM using LangChain's OpenAI wrapper."""
        self.secondary_llm = OpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),  # Ensure this is set in your .env file
            temperature=0.7,
            max_tokens=2048,
        )
        print("Secondary LLM (ChatGPT-4) configured using LangChain.")

    def _setup_embeddings(self):
        """Initialize Google AI SDK for embeddings."""
        genai.configure(api_key=self.google_api_key)
        self.emb_model = self.emb_model or "models/text-embedding-004"  # fallback if env var not set
        
        # Test the embeddings setup
        try:
            test_response = genai.embed_content(
                model=self.emb_model,
                content="Test embedding setup"
            )
            print(f"Embeddings setup successful. Vector dimension: {len(test_response['embedding'])}")
        except Exception as e:
            print(f"Error setting up embeddings: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> list:
        """
        Generate embeddings for given text using Google API.
        
        Args:
            text (str): The text to generate embeddings for
            
        Returns:
            list: The embedding vector
        """
        print("\nGenerating embeddings with Google Gemini: ", self.emb_model)
        try:
            response = genai.embed_content(
                model=self.emb_model,
                content=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def _setup_milvus(self):
        """Initialize Milvus vector store connection."""
        try:
            print(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}...")
            
            # First, establish direct Milvus connection
            from pymilvus import connections, Collection, utility
        
            connections.connect(db_name=self.milvus_db, host=self.milvus_host, port=self.milvus_port)
            
            # Define embedding function
            def embedding_function(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [self.generate_embedding(text) for text in texts]
            
            # Initialize Milvus through LangChain
            self.milvus = Milvus(
                embedding_function=embedding_function,
                collection_name=self.collection_name,
                connection_args={
                    "host": self.milvus_host,
                    "port": self.milvus_port
                }
            )
            
            # Get direct access to the collection
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
            
            print(f"Connected to Milvus collection: {self.collection_name}")
            print(f"Number of entities: {self.collection.num_entities}")
            
            # Setup retriever
            self.retriever = self.milvus.as_retriever(
                search_kwargs={
                    "k": 3,
                    "search_type": "similarity",
                    "param": {"metric_type": "L2"}
                }
            )
            
        except Exception as e:
            print(f"Error setting up Milvus: {str(e)}")
            raise

    def _setup_retrieval_chain(self):
        """Initialize the RAG chain with a custom prompt."""
        # Define the prompt for combining documents
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Use the following pieces of context to answer the user's question. 
            If you don't know the answer based on the context, use your knowledgebase or search the web.
            
            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_variable_name="context",
        )

        # Create the retrieval chain
        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )

    def _setup_chat_memory(self):
        """Initialize chat memory and conversation chain with proper history handling."""
        # Initialize chat history
        self.chat_history = []
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="input"
        )
        
        # Create the chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to a knowledge base. 
            Use the provided context to answer questions accurately.
            If you don't know the answer based on the context, use your knowledgebase or search the University of Chicago data science website.
            
            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create the runnable chain
        chain = (
            {
                "context": lambda x: "\n".join([doc.page_content if hasattr(doc, "page_content") else doc for doc in x.get("context", [])]),
                "input": lambda x: x["input"],
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"]
            }
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Initialize the conversation chain
        self.conversation_chain = chain
        
    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with proper memory handling, returning a document containing specific keywords if available.
        """
        try:
            # Get relevant documents with similarity scores
            search_results = self.similarity_search(question)
            
            # Define a maximum threshold for relevance
            max_relevance_threshold = 0.3  

            # Check if Milvus found relevant results
            if search_results:
                # Find the best match (lowest L2 distance)
                best_doc, best_score = min(search_results, key=lambda x: x[1])

                # Check if the best match is within the max relevance threshold
                if best_score <= max_relevance_threshold:
                    print("Relevant document found in Milvus.")
                    context = [best_doc]
                else:
                    # If the score is above the threshold, treat it as irrelevant and fallback to web search
                    print("No relevant documents within threshold in Milvus, attempting web search...")
                    context = [self.perform_web_search(question)]
            else:
                # No results found in Milvus, proceed to web search
                print("No results found in Milvus, attempting web search...")
                context = [self.perform_web_search(question)]

            # Prepare input for conversation chain
            chain_input = {
                "input": question,
                "context": context,
                "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
            }

            # Execute the conversation chain
            answer = self.conversation_chain.invoke(chain_input)
            print("### LLM Answer: {}".format(answer))

            # Save question and answer to memory if memory usage is enabled
            if use_memory:
                self.memory.save_context({"input": question}, {"answer": answer})

            return {
                "answer": answer,
                "source_document": {
                    "content": context[0],
                    "metadata": {"url": best_doc.metadata.get("url")} if search_results else {"url": "Web search"},
                    "similarity_score": best_score if search_results else None
                }
            }

        except Exception as e:
            print(f"Error during query: {str(e)}")
            traceback.print_exc()
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_document": None
            }
            
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in Milvus using direct collection access.
        """
        try:
            print(f"\nExecuting similarity search for query: '{query}'")
            
            # Generate embedding
            query_embedding = self.generate_embedding(query)
            print(f"Generated embedding dimension: {len(query_embedding)}")
            
            # Use direct collection access for search
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Execute search using the collection directly
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",  # Make sure this matches your schema
                param=search_params,
                limit=k,
                output_fields=["content", "url"]
            )
            
            if results:
                print("Query returned results!")
            
            # Convert to Document format
            documents_with_scores = []
            for hits in results:
                for hit in hits:
                    try:
                        # Debug the hit object
                        print("\nRaw hit data:")
                        print(f"Distance: {hit.distance}")
                        print(f"ID: {hit.id}")
                        
                        # Get entity fields using proper attribute access
                        entity = hit.entity
                        
                        # Create document using the entity data
                        doc = Document(
                            page_content=str(entity.content),  # Access as attribute
                            metadata={
                                'id': hit.id,
                                'score': hit.distance,
                                'url': str(entity.url)  # Access as attribute
                            }
                        )
                        documents_with_scores.append((doc, hit.distance))
                        
                        # Debug output
                        print(f"\nProcessed result:")
                        print(f"Score: {hit.distance}")
                        print(f"Content: {doc.page_content}")
                        print(f"URL: {doc.metadata['url']}")
                        
                    except AttributeError as e:
                        print(f"Attribute error processing hit: {e}")
                        print(f"Available entity attributes: {dir(hit.entity)}")
                        continue
                    except Exception as e:
                        print(f"Error processing hit: {e}")
                        print(f"Hit structure: {dir(hit)}")
                        continue
            
            print(f"\nFound {len(documents_with_scores)} valid results")
            return documents_with_scores
            
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def verify_search_result(self, query: str = "test query"):
        """
        Verify search functionality with detailed debugging.
        """
        try:
            print("\nVerifying search functionality...")
            
            # Test embedding generation
            embedding = self.generate_embedding(query)
            print(f"✓ Successfully generated embedding of dimension {len(embedding)}")
            
            # Test search
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=1,
                output_fields=["content", "url"]
            )
            
            if results and len(results) > 0:
                hit = results[0][0]
                print("\n✓ Search successful!")
                print(f"Sample result:")
                print(f"Distance: {hit.distance}")
                print(f"ID: {hit.id}")
                print(f"Entity structure: {hit.entity}")
            else:
                print("✗ No results found")
                
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            traceback.print_exc()
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared")
    
    def test_embedding(self, query):
        # Generate the embedding (returns an EmbeddingDict object)
        embedding_response = genai.embed_content(model=self.emb_model, content=query)

        # Extract the embedding, which is already a list of floats
        embedding = embedding_response["embedding"]

        # Debug: Print the embedding type and sample values
        print(f"Embedding type: {type(embedding)}, Length: {len(embedding)}")
        print(f"First 5 values of embedding: {embedding[:5]}")
        print("Full embedding:", embedding)

    def get_collection_info(self):
        """
        Get information about the Milvus collection.
        """
        try:
            print("\nCollection Information:")
            print(f"Collection name: {self.collection.name}")
            print(f"Number of entities: {self.collection.num_entities}")
            
            print("\nSchema information:")
            for field in self.collection.schema.fields:
                print(f"Field: {field.name}, Type: {field.dtype}")
            
            print("\nIndex information:")
            indexes = self.collection.indexes
            for index in indexes:
                print(f"Index: {index}")
                
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            traceback.print_exc()


    def perform_web_search(self, query: str) -> str:
        """
        Perform a web search using Google Custom Search JSON API for cases where no relevant documents are found.
        Args:
            query (str): The query to search for.
        Returns:
            str: The top search result or a summary of the results.
        """
        try:
            # Use Google Custom Search JSON API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "cx": os.getenv("GOOGLE_SEARCH_CX"),  # Google custom Search Engine ID
                "key": os.getenv("GOOGLE_API_KEY"),   # Google API key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
        
            search_results = response.json()
            print("Google search response: ",search_results)
            # Extract and return the first result's snippet
            if "items" in search_results:
                top_result = search_results["items"][0]["snippet"]
                return top_result
            else:
                return "No relevant results found online."

        except Exception as e:
            print(f"Error during Google web search: {str(e)}")
            return "I couldn't find relevant information online either."

    

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = LangChainRAGSystem()
    #rag_system.get_collection_info()
    sim_search_query = "capstone project"
    
    # create vector embeddings
   # print("\nGenerating embeddings...")
   # rag_system.generate_embedding(sim_search_query)
    
   # rag_system.verify_search_result("capstone project")

    # Example: Similarity search only
    # print("\nSimilarity Search Results:")
    # search_results = rag_system.similarity_search(sim_search_query)
    # for doc, score in search_results:
    #     print(f"- Document: {doc.page_content}")
    #     print(f"  Similarity Score: {score}")
    
    # Test the memory-enabled query
    response = rag_system.query(
        "Can I set up an advising appointment with the enrollment management team?", 
        use_memory=True
    )
    print("First Response:", response['answer'])

    # Ask a follow-up question
    response = rag_system.query(
        "Where can I mail my official transcripts?", 
        use_memory=True
    )
    print("Follow-up Response:", response['answer'])

    # Clear memory if needed
    rag_system.clear_memory()