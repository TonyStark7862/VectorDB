import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vector_db.log")
    ]
)
logger = logging.getLogger("VectorDB")

class VectorDatabase:
    """
    A vector database implementation using Faiss for indexing and retrieval,
    with pickle for persistence.
    """
    
    def __init__(self, 
                 dimension: int = 768, 
                 index_type: str = "L2",
                 index_path: str = "faiss_index",
                 metadata_path: str = "metadata.pickle"):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of index to use ('L2', 'IP', 'HNSW', etc.)
            index_path: Path to save/load the Faiss index
            metadata_path: Path to save/load the metadata
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = {}
        self.index = None
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the Faiss index based on the specified type."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            else:
                logger.info("Creating new index")
                if self.index_type == "L2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == "IP":
                    self.index = faiss.IndexFlatIP(self.dimension)
                elif self.index_type == "HNSW":
                    # More efficient for large datasets
                    self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                elif self.index_type == "IVF":
                    # For very large datasets with approximate nearest neighbor search
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    nlist = 100  # number of clusters
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                    self.index.train(np.zeros((1000, self.dimension), dtype=np.float32))
                else:
                    logger.warning(f"Unknown index type: {self.index_type}, using L2")
                    self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = {
                    "documents": {},
                    "total_chunks": 0,
                    "file_to_chunk_ids": {},
                    "document_types": {}
                }
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            raise

    def add_embeddings(self, 
                     embeddings: np.ndarray, 
                     chunk_contents: List[str],
                     chunk_metadata: List[Dict],
                     file_id: str,
                     document_type: str):
        """
        Add embeddings and corresponding chunk content to the index.
        
        Args:
            embeddings: Numpy array of embeddings, shape (n, dimension)
            chunk_contents: List of chunk text contents
            chunk_metadata: List of metadata dicts for each chunk
            file_id: Unique identifier for the file
            document_type: Type of document (pdf, csv, etc.)
        
        Returns:
            List of IDs assigned to the chunks
        """
        try:
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
            # Ensure embeddings are float32 (required by Faiss)
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
                
            # Get the next available IDs
            start_id = self.metadata["total_chunks"]
            n_vectors = embeddings.shape[0]
            
            # Add to Faiss index
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            self.index.add(embeddings)
            
            # Track the range of IDs for this file
            chunk_ids = list(range(start_id, start_id + n_vectors))
            
            # Update metadata
            for i, chunk_id in enumerate(chunk_ids):
                self.metadata["documents"][chunk_id] = {
                    "content": chunk_contents[i],
                    "metadata": chunk_metadata[i],
                    "file_id": file_id
                }
            
            # Update file to chunk mapping
            if file_id not in self.metadata["file_to_chunk_ids"]:
                self.metadata["file_to_chunk_ids"][file_id] = []
            self.metadata["file_to_chunk_ids"][file_id].extend(chunk_ids)
            
            # Update document type tracking
            self.metadata["document_types"][file_id] = document_type
            
            # Update total
            self.metadata["total_chunks"] += n_vectors
            
            logger.info(f"Added {n_vectors} vectors for file {file_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise

    def search(self, 
              query_embedding: np.ndarray, 
              k: int = 5,
              file_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for the most similar vectors in the index.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            file_ids: Optional list of file IDs to restrict search to
        
        Returns:
            List of dicts with retrieved chunks and their metadata
        """
        try:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
                
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Ensure the embedding is float32
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
                
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # If file_ids is provided, create a filtered index
            if file_ids and len(file_ids) > 0:
                # Get all chunk IDs for the specified files
                chunk_ids = []
                for file_id in file_ids:
                    if file_id in self.metadata["file_to_chunk_ids"]:
                        chunk_ids.extend(self.metadata["file_to_chunk_ids"][file_id])
                
                if not chunk_ids:
                    logger.warning(f"No chunks found for file_ids: {file_ids}")
                    return []
                
                # Create a subset index
                subset_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                vectors = np.zeros((len(chunk_ids), self.dimension), dtype=np.float32)
                
                for i, chunk_id in enumerate(chunk_ids):
                    # Reconstruct the vector from the original index
                    self.index.reconstruct(chunk_id, vectors[i])
                
                subset_index.add_with_ids(vectors, np.array(chunk_ids))
                distances, indices = subset_index.search(query_embedding, min(k, len(chunk_ids)))
            else:
                # Search the full index
                distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            results = []
            for i in range(indices.shape[1]):
                chunk_id = int(indices[0, i])
                if chunk_id >= 0 and chunk_id in self.metadata["documents"]:  # -1 indicates no match found
                    doc = self.metadata["documents"][chunk_id]
                    results.append({
                        "chunk_id": chunk_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "file_id": doc["file_id"],
                        "distance": float(distances[0, i]),
                        "similarity": 1.0 / (1.0 + float(distances[0, i]))  # Convert distance to similarity
                    })
            
            logger.info(f"Retrieved {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise

    def delete_file(self, file_id: str) -> bool:
        """
        Delete all chunks associated with a file.
        
        Args:
            file_id: ID of the file to delete
        
        Returns:
            Success status
        """
        try:
            if file_id not in self.metadata["file_to_chunk_ids"]:
                logger.warning(f"File {file_id} not found in index")
                return False
                
            # Get chunk IDs for this file
            chunk_ids = self.metadata["file_to_chunk_ids"][file_id]
            
            # Currently, Faiss doesn't support efficient deletion
            # So we need to rebuild the index without these vectors
            logger.info(f"Rebuilding index to remove file {file_id} with {len(chunk_ids)} chunks")
            
            # Create a new index of the same type
            new_index = None
            if self.index_type == "L2":
                new_index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IP":
                new_index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "HNSW":
                new_index = faiss.IndexHNSWFlat(self.dimension, 32)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = 100
                new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                
            # Create a set of chunk IDs to remove for fast lookup
            chunk_ids_to_remove = set(chunk_ids)
            
            # Collect vectors to keep
            keep_vectors = []
            keep_ids = []
            
            for i in range(self.metadata["total_chunks"]):
                if i not in chunk_ids_to_remove and i in self.metadata["documents"]:
                    vector = np.zeros((1, self.dimension), dtype=np.float32)
                    self.index.reconstruct(i, vector[0])
                    keep_vectors.append(vector)
                    keep_ids.append(i)
            
            # If we have vectors to keep
            if keep_vectors:
                keep_vectors = np.vstack(keep_vectors)
                
                # For IVF index, we need to train first
                if self.index_type == "IVF":
                    new_index.train(keep_vectors)
                
                new_index.add(keep_vectors)
                
                # Update metadata
                new_metadata = {
                    "documents": {},
                    "total_chunks": len(keep_ids),
                    "file_to_chunk_ids": {},
                    "document_types": {k: v for k, v in self.metadata["document_types"].items() if k != file_id}
                }
                
                # Copy over documents we're keeping
                for i, old_id in enumerate(keep_ids):
                    new_metadata["documents"][i] = self.metadata["documents"][old_id].copy()
                    
                    # Update file to chunk mapping
                    file_id_for_chunk = new_metadata["documents"][i]["file_id"]
                    if file_id_for_chunk not in new_metadata["file_to_chunk_ids"]:
                        new_metadata["file_to_chunk_ids"][file_id_for_chunk] = []
                    new_metadata["file_to_chunk_ids"][file_id_for_chunk].append(i)
                
                # Replace old index and metadata
                self.index = new_index
                self.metadata = new_metadata
            else:
                # No vectors left, reset everything
                self._initialize_index()
            
            logger.info(f"Successfully removed file {file_id} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False

    def save(self):
        """Save the index and metadata to disk."""
        try:
            # Create directories if they don't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save the metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
            
    def load(self):
        """Load the index and metadata from disk."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded index with {self.index.ntotal} vectors from {self.index_path}")
                return True
            else:
                logger.warning(f"Index files not found at {self.index_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
            
    def get_stats(self):
        """Get statistics about the index."""
        return {
            "total_chunks": self.metadata["total_chunks"],
            "total_files": len(self.metadata["file_to_chunk_ids"]),
            "document_types": self.metadata["document_types"],
            "index_type": self.index_type,
            "dimension": self.dimension
        }
