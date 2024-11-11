import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import traceback
import requests
# Core LangChain imports

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_community.chat_models import ChatOpenAI


# Memory and callbacks
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers import LangChainTracer

# LangSmith
from langsmith import Client

# Chain types for RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import google.generativeai as genai
from prompt_security import PromptInjection


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
       # self._setup_embeddings()
        self._setup_milvus()
        self.configure_gemini()
        self._setup_llm()
        self._setup_retrieval_chain()
        self._setup_chat_memory()
        #self._setup_secondary_llm()
        

    def _setup_llm(self):
        """Initialize the primary LLM, Google Gemini."""
        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY"),  
            temperature=0.7,
            max_tokens=2048,
        )
        print("Primary LLM (ChatGPT4) configured.")
    
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
 
    def generate_embedding(self, text: str) -> list:
        """
        Generate and return an embedding vector for a given text input.
        """
        try:
            # Initialize the OpenAI embedding model
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
            
            # Generate the embedding for the provided text
            embedding_vector = embeddings.embed_query(text)  # This returns the embedding vector as a list of floats
            print(f"Embedding generated successfully. Dimension: {len(embedding_vector)}")

            return embedding_vector
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
            ("system", """You are a chatbot for University of Chicagos Applied Data Science program.
            Use the following pieces of context to answer the user's question regarding this program. 
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
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant for the University of Chicago's Applied Data Science program.
                Use the provided context to answer the user's question about this program. If the context includes specific course titles or program details, 
                list them explicitly. Only recommend checking the official website if essential details are missing in the context.
                
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
    

    def extract_keywords(self, query: str):
        keywords = self.kw_model.extract_keywords(query, top_n=2, keyphrase_ngram_range=(1, 2))
        return [kw[0] for kw in keywords]  # Return only keyword strings
    
    
    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with proper memory handling, using ChatGPT-4 as fallback if needed.
        """
        try:
            promptinjection = PromptInjection()
            
            # Detect prompt injection first to make sure prompt is safe for handling
            injection_attempt = promptinjection.detect_prompt_injection(question)
            
            if not injection_attempt == "prompt_safe":
                return injection_attempt
            
            # Get relevant documents with similarity scores
            #search_results = self.similarity_search(question)
            search_results = self.hybrid_search(question)
            
            for doc, score in search_results:
                print("-"*100)
                print(f"Content: {doc.page_content}\nScore: {score}\nURL: {doc.metadata['url']}\n")
                print("-"*100)
                
            # Get the top results
            context_size = 3
            # Check if Milvus found relevant results
            if search_results:
                # Sort the search results by score (ascending for L2 distance)
                sorted_results = sorted(search_results, key=lambda x: x[1])

                # Select the top `context_size` chunks and extract text content
                selected_chunks = [doc.page_content for doc, score in sorted_results[:context_size]]
                urls = [doc.metadata.get("url", "") for doc, score in sorted_results[:context_size]]

                # Debug information to show selected chunks and scores
                for idx, (doc, score) in enumerate(sorted_results[:context_size], 1):
                    print(f"Top Result {idx}:")
                    print(f"Score: {score}")
                    print(f"Content Preview: {doc.page_content[:200]}...\n")

                # Aggregate the selected chunks to form the final context
                aggregated_context = "\n".join(selected_chunks)         
                # if best_score >= min_relevance_threshold:  # Check if best score meets or exceeds the threshold
                print("Relevant document found in Milvus")
                print("Context being provided to LLM: {}".format(aggregated_context))
                context = [aggregated_context]
                
                # else:
                #     print("No relevant documents within threshold in Milvus, attempting web search...")
                #     context = [self.perform_web_search(question)]
            # else:
            #     print("No results found in Milvus, attempting web search...")
            #     context = [self.perform_web_search(question)]

            # Prepare input for conversation chain
            chain_input = {
                "input": question,
                "context": context,
                "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
            }

            # Execute the conversation chain with primary LLM
            answer = self.conversation_chain.invoke(chain_input)
            print("### Primary LLM Answer:", answer)

            # # Check if the primary LLM answer is empty or unsatisfactory
            # if not answer or "I don't know" or "I couldn't" in answer:  
            #     print("Primary LLM returned no result, using ChatGPT-4 as fallback.")
            #     # Call ChatGPT-4 as fallback using LangChain OpenAI wrapper
            #     answer = self._query_chatgpt4_with_langchain(question, context)

            #     if not answer or "I don't know" or "I couldn't" in answer:
            #         print("ChatGPT-4 also failed to provide an answer.")
            #         answer = "I couldn't find an answer to your question. Please check the link below for more information"
            
            # Save question and answer to memory if memory usage is enabled
            if use_memory:
                self.memory.save_context({"input": question}, {"answer": answer})

            return {
                "answer": answer,
                "source_document": {
                    "content": context[0],
                    "metadata": {"urls": urls},
                }
            }

        except Exception as e:
            print(f"Error during query: {str(e)}")
            traceback.print_exc()
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_document": None
            }

    # Helper method to query ChatGPT-4 via LangChain
    def _query_chatgpt4_with_langchain(self, question: str, context: List[str]) -> str:
        """Query ChatGPT-4 for a response via LangChain's ChatOpenAI wrapper."""
        try:
            # Prepare the messages for ChatGPT-4, including context if available
            context_text = "\n\n".join([str(c) for c in context])
            messages = [
                ( "system", "You are a helpful assistant."),
                ("user", f"Context:\n{context_text}\n\nQuestion: {question}")
            ]

            # Use LangChain's ChatOpenAI wrapper to get the response from ChatGPT-4
            response = self.secondary_llm.invoke(messages)
            print(response)
            chatgpt4_answer = response['choices'][0]['message']['content']
            print("### ChatGPT-4 Answer via LangChain:", chatgpt4_answer)
            return chatgpt4_answer

        except Exception as e:
            print(f"Error querying ChatGPT-4 via LangChain: {str(e)}")
            return "I couldn't retrieve an answer from ChatGPT-4 either."
            
    # def similarity_search(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
    #     """
    #     Perform a similarity search in Milvus using direct collection access.
    #     """
    #     try:
    #         print(f"\nExecuting similarity search for query: '{query}'")
            
    #         # Generate embedding
    #         query_embedding = self.generate_embedding(query)
    #         print(f"Generated embedding dimension: {len(query_embedding)}")
            
    #         # Use direct collection access for search
    #         search_params = {
    #             "metric_type": "COSINE",
    #             "params": {"nprobe": 32}
    #         }
            
    #         # Execute search using the collection directly
    #         results = self.collection.search(
    #             data=[query_embedding],
    #             anns_field="embedding",  # Make sure this matches your schema
    #             param=search_params,
    #             limit=k,
    #             output_fields=["content", "url"]
    #         )
            
    #         if results:
    #             print("Query returned results!")
            
    #         # Convert to Document format
    #         documents_with_scores = []
    #         for hits in results:
    #             for hit in hits:
    #                 try:
    #                     # Debug the hit object
    #                     print("\nRaw hit data:")
    #                     print(f"Distance: {hit.distance}")
    #                     print(f"ID: {hit.id}")
                        
    #                     # Get entity fields using proper attribute access
    #                     entity = hit.entity
                        
    #                     # Create document using the entity data
    #                     doc = Document(
    #                         page_content=str(entity.content),  # Access as attribute
    #                         metadata={
    #                             'id': hit.id,
    #                             'score': hit.distance,
    #                             'url': str(entity.url)  # Access as attribute
    #                         }
    #                     )
    #                     documents_with_scores.append((doc, hit.distance))
                        
    #                     # Debug output
    #                     print(f"\nProcessed result:")
    #                     print(f"Score: {hit.distance}")
    #                     print(f"Content: {doc.page_content}")
    #                     print(f"URL: {doc.metadata['url']}")
                        
    #                 except AttributeError as e:
    #                     print(f"Attribute error processing hit: {e}")
    #                     print(f"Available entity attributes: {dir(hit.entity)}")
    #                     continue
    #                 except Exception as e:
    #                     print(f"Error processing hit: {e}")
    #                     print(f"Hit structure: {dir(hit)}")
    #                     continue
            
    #         print(f"\nFound {len(documents_with_scores)} valid results")
    #         return documents_with_scores
            
    #     except Exception as e:
    #         print(f"Error during similarity search: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return []

    def expand_keywords(self, keywords):
        """
        Expand keywords to include both singular and plural forms
        
        Args:
            keywords (list): List of keywords/keyphrases from the extraction model
            
        Returns:
            list: Expanded list of keywords including singular/plural forms
        """
        expanded = set()
        
        for keyphrase in keywords:
            # Add original keyphrase
            expanded.add(keyphrase)
            
            # Handle individual words in the keyphrase
            words = keyphrase.split()
            for word in words:
                # Add original word
                expanded.add(word)
                
                # Add singular if word ends in 's'
                if word.endswith('s') and len(word) > 3:
                    singular = word[:-1]
                    expanded.add(singular)
                # Add plural if word doesn't end in 's'
                elif not word.endswith('s'):
                    plural = word + 's'
                    expanded.add(plural)
                    
        return list(expanded)
    
    def rank_result(self, content: str, keywords: List[str]) -> float:
        """
        Calculate relevance score for a result
        
        Args:
            content (str): Document content
            keywords (List[str]): Original keywords/phrases
            
        Returns:
            float: Relevance score
        """
        score = 0
        content = content.lower()
        
        # Check exact phrase matches (highest priority)
        for phrase in keywords:
            if len(phrase.split()) > 1 and phrase.lower() in content:
                score += 10  # Higher weight for exact phrases
        
        # Check individual word matches
        for keyword in keywords:
            words = keyword.lower().split()
            for word in words:
                if word in content:
                    score += 1
        
        return score

    def hybrid_search(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """
        Hybrid search with deduplication and detailed debug output
        """
        try:
            print(f"\nExecuting hybrid search for query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Extract keywords and build expression
            # keywords = self.extract_keywords(query)
            # expanded_keywords = self.expand_keywords(keywords)
            # filter_expression = " or ".join([f'content like "%{k}%"' for k in expanded_keywords])
            
            # print(f"Extracted Keywords: {keywords}")
            # print("Filter expression: ", filter_expression)
            
            search_params = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 10,
                }
            }
        
            # Execute search with increased limit to account for duplicates
            search_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k * 5,  # Double the limit to account for duplicates
                output_fields=["metadata"]  
            )
        
            print("\nDebug: Raw Search Results")
            print("-" * 80)

            # Process and debug print all results with deduplication
            documents_with_scores = []
            seen_hashes = set()

            for hits in search_results:
                for hit in hits:
                    try:
                        # Directly access fields in `metadata`
                        metadata = hit.entity.metadata  # Access metadata directly
                        content = metadata.get("content", "")  # Safely access 'content' field
                        url = metadata.get("url", "")  # Safely access 'url' field
                        score = hit.distance

                        print(f"\nContent: {content[:200]}...")  # Display a preview of the content
                        print(f"Score: {score}")
                        print(f"URL: {url}")
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'score': score,
                                'url': url
                            }
                        )
                        documents_with_scores.append((doc, score))

                    except Exception as e:
                        print(f"Error processing hit: {e}")
                        continue

            return documents_with_scores
        
        
        #     print("\nDebug: Raw Search Results")
        #     print("-" * 80)
            
        #     # Process and debug print all results with deduplication
        #     documents_with_scores = []
        #     seen_hashes = set()
            
        #     for hits_idx, hits in enumerate(results):
        #         print(f"\nResult Set {hits_idx + 1}:")
        #         for hit_idx, hit in enumerate(hits):
        #             try:
        #                 entity = hit.entity
        #                 content = str(entity.content)
        #                 url = str(entity.url)
        #                 chunk_hash = str(entity.chunk_hash)
                        
        #                 # Skip if we've seen this content before
        #                 if chunk_hash in seen_hashes:
        #                     print(f"\nSkipping duplicate content (hash: {chunk_hash[:8]})")
        #                     continue
                        
        #                 seen_hashes.add(chunk_hash)
                        
        #                 print(f"\nDocument {hit_idx + 1}:")
        #                 print(f"Distance Score: {hit.distance}")
        #                 print(f"URL: {url}")
        #                 print(f"Content Hash: {chunk_hash[:8]}")  # Show first 8 chars of hash
        #                 print(f"Content Preview: {content[:200]}...")
                     
                        
        #                 doc = Document(
        #                     page_content=content,
        #                     metadata={
        #                         'id': hit.id,
        #                         'score': hit.distance,
        #                         'url': url,
        #                         'chunk_hash': chunk_hash  # Add hash to metadata
        #                     }
        #                 )
        #                 documents_with_scores.append((doc, hit.distance))
                        
        #                 # Break if we have enough unique results
        #                 if len(documents_with_scores) >= k:
        #                     break
                        
        #             except Exception as e:
        #                 print(f"Error processing hit {hit_idx}: {e}")
        #                 continue
                
        #         if len(documents_with_scores) >= k:
        #             break
            
        #     print("\nFinal Results Summary:")
        #     print(f"Total unique results: {len(documents_with_scores)}")
            
        #     # Sort by score
        #     documents_with_scores.sort(key=lambda x: x[1])
            
        #     print("\nTop 3 Unique Results After Sorting:")
        #     for idx, (doc, score) in enumerate(documents_with_scores[:3], 1):
        #         print(f"\nTop Result {idx}:")
        #         print(f"Score: {score}")
        #         print(f"URL: {doc.metadata['url']}")
        #         print(f"Hash: {doc.metadata['chunk_hash'][:8]}")
        #         print(f"Content Preview: {doc.page_content[:200]}...")
            
        #     return documents_with_scores
                
        except Exception as e:
            print(f"Error during hybrid search: {str(e)}")
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