from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from openai import OpenAI
import numpy as np
import os
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
import tiktoken

class MilvusLoader:
    def __init__(self, host="35.184.233.137", port="19530", collection_name="ms_data_science_v1", dim=3072, env_path: str = ".env"):
        load_dotenv(env_path)
        # Initialize OpenAI
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.embedding_model = "text-embedding-3-large"  # Specify OpenAI embedding model
        self.client = OpenAI()
        # Milvus connection and parameters
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.connect()
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def connect(self):
        # Connect to Milvus server
        connections.connect(db_name=os.getenv('MILVUS_DB'), host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def create_collection(self):
        # Check if the collection already exists
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists. Appending data.")
            self.collection = Collection(name=self.collection_name)
        else:
            # Define collection schema and create a new collection if it doesn't exist
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, "Capstone Project Documents Embeddings")
            
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")

        
    def load_data(self, file_path):
        # Load JSON data from file
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, Markdown-specific characters, and redundant sections."""
        text = re.sub(r'https?://\S+', '', text)        # Remove URLs
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)       # Remove Markdown links
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)      # Remove Markdown images
        text = re.sub(r'[#*>\-]', '', text)              # Remove Markdown symbols
        text = re.sub(r'\s+', ' ', text).strip()         # Reduce whitespace
        return text

    def split_text(self, text: str, max_tokens: int = 300, overlap: int = 20) -> List[str]:
        """Split text into chunks, ending at full stops and under a token limit."""
        sentences = re.split(r'\.\s+', text)  # Split on full stops followed by whitespace
        chunks = []
        current_chunk = []

        for sentence in sentences:
            # Add a period to each sentence if it was split
            sentence = sentence.strip() + '.'
            
            # Test if adding the next sentence exceeds max_tokens
            chunk_text = " ".join(current_chunk + [sentence])
            token_count = len(self.tokenizer.encode(chunk_text))

            if token_count <= max_tokens:
                # Add sentence to the current chunk if within token limit
                current_chunk.append(sentence)
            else:
                # Finalize current chunk and start a new one
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = [sentence]  # Start new chunk with the current sentence

        # Add any remaining sentences as the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        # Add overlap by including the last `overlap` tokens from the previous chunk in the next one
        final_chunks = []
        for i in range(len(chunks)):
            chunk_tokens = self.tokenizer.encode(chunks[i])
            if i > 0:  # Start each chunk with the last `overlap` tokens of the previous chunk
                overlap_tokens = self.tokenizer.encode(chunks[i - 1])[-overlap:]
                chunk_tokens = overlap_tokens + chunk_tokens
            final_chunks.append(self.tokenizer.decode(chunk_tokens))

        return final_chunks


    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding from the response
            embedding = response.data[0].embedding  # Access attributes instead of subscript notation
            
            # Normalize embedding
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                return (embedding_array / norm).tolist()
            return None

        except Exception as e:
            print(f"✗ Error generating embedding: {str(e)}")
            return None

    def insert_data(self, data: List[Dict]):
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized. Please run create_collection first.")

        embeddings = []
        metadata = []
        for item in data:
            raw_text = item.get("markdown", "")
            text = self.clean_text(raw_text)
            
            # Split into chunks if text is too long
            chunks = self.split_text(text, max_tokens=300)
            for chunk in chunks:
                embedding = self.generate_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)
                    
                    # Include chunk content in metadata
                    metadata.append({
                        "content": chunk,
                        **item.get("metadata", {})  # Merge additional metadata if needed
                    })
                else:
                    print(f"✗ Skipping entry due to failed embedding generation for text chunk: {chunk[:30]}...")

        if embeddings:  # Insert only if there are valid embeddings
            self.collection.insert([embeddings, metadata])
            print(f"Inserted {len(embeddings)} records into '{self.collection_name}'.")
        else:
            print("✗ No valid embeddings to insert.")

    def create_index(self):
        # Create index for efficient similarity search
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized. Please run create_collection first.")

        # Define index parameters with metric type
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",  # Use L2 (Euclidean) for normalized vectors; use "IP" if not normalized
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        print(f"Index created and loaded for collection '{self.collection_name}'.")

    def process_and_insert(self, file_path):
        # Full pipeline to load JSON, process, and insert into Milvus
        data = self.load_data(file_path)
        self.insert_data(data)
        self.create_index()
        print("Data processing and insertion complete.")

    def verify_collection(self):
        collections = utility.list_collections()
        print("Collections in Milvus:", collections)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for documents in the Milvus collection that are most similar to the query.
        
        Args:
            query (str): The text query to search for similar documents.
            top_k (int): Number of top matches to retrieve.

        Returns:
            List[Tuple[Dict, float]]: A list of tuples containing metadata and similarity score.
        """
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            print("Failed to generate embedding for the query.")
            return []

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"]
        )

        matched_documents = []
        for hits in results:
            for hit in hits:
                try:
                    metadata = hit.entity.metadata  # Access metadata
                    score = hit.distance

                    # Append metadata and score to the result list
                    matched_documents.append((metadata, score))

                except Exception as e:
                    print(f"Error processing hit: {e}")
                    continue

        # Return the top-k matched documents with their scores
        return matched_documents
    
    def print_matches(self, matches: List[Tuple[Dict, float]]):
        print(f"Top {len(matches)} matches found:")
        for i, (metadata, score) in enumerate(matches, 1):
            print(f"Match {i}:")
            print(f"Similarity Score: {score:.4f}")
            print("Metadata:")
            print(metadata)
            print()
    
# Usage
milvus_loader = MilvusLoader()
milvus_loader.create_collection()
milvus_loader.process_and_insert("data/results_6.json")
milvus_loader.verify_collection()
matched_docs = milvus_loader.search("What are the courses offered by the program?", top_k=3)
milvus_loader.print_matches(matched_docs)