import pandas as pd
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, exceptions, utility
import google.generativeai as genai
from urllib.parse import urljoin
from dotenv import load_dotenv
import os
from google.api_core import retry

class MilvusEmbeddingHandler:
    def __init__(self, host='127.0.0.1', port='19530', collection_name='ms_applied_data_science', env_path=".env"):
        load_dotenv(env_path)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None

        # Retrieve and configure API key
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.configure_genai()

        # Initialize the Gemini model
        self.model = 'models/text-embedding-004'

    def configure_genai(self):
        """Configure the Gemini API with the provided API key."""
        if not self.api_key:
            raise ValueError("API Key is missing. Please provide a valid API key.")
        genai.configure(api_key=self.api_key)
        print("Gemini API configured successfully.")

    def connect_to_milvus(self):
        """Connect to Milvus."""
        connections.connect(db_name="genai", host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def create_collection(self):
        """Create a collection if it doesn't already exist."""
        
        self.connect_to_milvus()
        
        if utility.has_collection(self.collection_name):  # Corrected function call
            print(f"Collection '{self.collection_name}' already exists.")
            self.collection = Collection(self.collection_name)
        else:
            # Define schema and create a new collection
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust dimension if needed
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512)
            ]
            schema = CollectionSchema(fields, "MS Applied Data Science Content Embeddings")
            self.collection = Collection(self.collection_name, schema)
            print(f"Collection '{self.collection_name}' created successfully.")



    def chunk_text(self, text, max_tokens=300):
        """Split text into manageable chunks, handling missing values gracefully."""
        if not isinstance(text, str):
            print("Warning: Encountered non-string content, skipping...")
            return []  # Return an empty list if the content is not a valid string

        words = text.split()
        return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    def load_data(self, csv_path):
        """Load scraped data from CSV."""
        return pd.read_csv(csv_path)
    
    def clean_data(self, data):
        """Clean the data by removing rows with non-string or NaN content."""
        print("Cleaning data: Removing rows with non-string or NaN content...")
        # Keep only rows where 'content' is a string and not NaN
        cleaned_data = data.dropna(subset=['content'])
        cleaned_data = cleaned_data[cleaned_data['content'].apply(lambda x: isinstance(x, str))]
        print(f"Cleaned data: {len(cleaned_data)} valid rows remaining.")
        return cleaned_data

    def insert_data(self, data):
        """Embed and insert data into Milvus."""
        if not self.collection:
            raise RuntimeError("Collection is not initialized. Call 'create_collection' first.")

        # Step 1: Clean the data
        cleaned_data = self.clean_and_validate_data(data)

        # Step 2: Generate embeddings
        contents, embeddings, urls = self.generate_embeddings(cleaned_data)

        # Step 3: Insert the data into Milvus
        if embeddings:
            self.insert_into_milvus(contents, embeddings, urls)
        else:
            print("No embeddings generated. Insertion into Milvus skipped.")

    def clean_and_validate_data(self, data):
        """Clean the input data and remove invalid rows."""
        print("Starting data cleaning...")
        cleaned_data = self.clean_data(data)
        print(f"Data cleaned: {len(cleaned_data)} valid rows found.")
        return cleaned_data

    def generate_embeddings(self, data):
        """Generate embeddings for the input data."""
        embeddings, contents, urls = [], [], []

        for index, row in data.iterrows():
            print(f"Processing row {index + 1}...")
            text_chunks = self.chunk_text(row['content'])

            if not text_chunks:
                print(f"No valid text chunks for row {index + 1}, skipping...")
                continue

            for chunk_index, chunk in enumerate(text_chunks):
                embedding = self.generate_single_embedding(chunk, index, chunk_index)
                if embedding:  # If embedding is valid
                    embeddings.append(embedding)
                    contents.append(chunk)
                    urls.append(row.get('url', ''))
        
        print(f"Total embeddings generated: {len(embeddings)}")
        return contents, embeddings, urls
    
    def generate_single_embedding(self, chunk, row_index, chunk_index):
        """Generate a single embedding, normalize it, and handle any errors."""
        try:
            # Generate the embedding (returns an EmbeddingDict object)
            embedding_response = genai.embed_content(model=self.model, content=chunk)

            # Extract the embedding, which is already a list of floats
            embedding = embedding_response["embedding"]

            # Debug: Print the embedding type and sample values
            print(f"Embedding type: {type(embedding)}, Length: {len(embedding)}")
            print(f"First 5 values of embedding: {embedding[:5]}")

            # Normalize the embedding using L2 norm
            normalized_embedding = self.normalize_embedding(embedding)
            
            return normalized_embedding

        except Exception as e:
            print(f"Error generating embedding for row {row_index + 1}, chunk {chunk_index + 1}: {e}")
            return None  # Return None if embedding generation fails

    def normalize_embedding(self, embedding):
        """Normalize the embedding to have unit length (L2 norm of 1)."""
        print("Normalizing embedding...")
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else embedding

    def insert_into_milvus(self, contents, embeddings, urls):
        """Insert the generated data into Milvus."""
        try:
            # Convert embeddings to float arrays
            embeddings = np.array(embeddings).astype(float)

            # Debug: Print data types and structure
            # print(f"Content type: {type(contents)}")
            # print(f"Embedding type: {type(embeddings)}")
            # print(f"URL type: {type(urls)}, Example: {urls[:1]}")

            # Create a list of dictionaries for Milvus batch insert
            data = [
                {"content": content, "embedding": embedding.tolist(), "url": url}
                for content, embedding, url in zip(contents, embeddings, urls)
            ]

            # Insert data into Milvus
            print("Inserting data into Milvus...")
            self.collection.insert(data)
            print(f"Successfully inserted {len(embeddings)} records into Milvus.")
        except Exception as e:
            print(f"Error inserting data into Milvus: {e}")
                       
    def verify_insertion(self):
        """Verify the number of records in the collection."""
        self.collection.load()
        print(f"Number of records in collection: {self.collection.num_entities}")

    def sample_run(self):
        """Test embedding generation and insertion with a single row."""
        if not self.collection:
            raise RuntimeError("Collection is not initialized. Call 'create_collection' first.")

        # Create a sample DataFrame with one row of content and URL
        sample_data = pd.DataFrame({
            'content': ["This is a sample text to test the Gemini embedding model."],
            'url': ["http://example.com"]
        })

        print("Running sample insertion test with one row...")
        try:
            # Use the same insert_data method with the sample DataFrame
            self.insert_data(sample_data)
        except Exception as e:
            print(f"Error during sample insertion: {e}")


# Usage
if __name__ == "__main__":
    # Initialize the handler with user data
    milvus_handler = MilvusEmbeddingHandler()

    milvus_handler.create_collection()
    
    # Load the scraped data
    scraped_data = milvus_handler.load_data('ms_applied_data_science_full_content_cleaned_data.csv')

    #milvus_handler.sample_run()
    
    # Insert the data into Milvus
    milvus_handler.insert_data(scraped_data)
    
    # Verify the insertion
    milvus_handler.verify_insertion()