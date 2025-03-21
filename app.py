import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import tempfile
import json
import pickle

# Import our custom modules
from vector_database import VectorDatabase
from document_processor import DocumentProcessor
from scoring import score_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("streamlit_app.log")
    ]
)
logger = logging.getLogger("StreamlitApp")

# Constants
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
EMBEDDING_DIM = 384  # Default for all-MiniLM-L6-v2

# Set page config
st.set_page_config(
    page_title="Vector DB RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
st.sidebar.title("Configuration")

# Models section
st.sidebar.header("Models")
embedding_model = st.sidebar.text_input(
    "Embedding Model Path",
    value=EMBEDDING_MODEL_PATH,
    help="Path to the embedding model (local or HuggingFace)"
)

# Retrieval options
st.sidebar.header("Retrieval Options")

use_reranker = st.sidebar.checkbox(
    "Use Re-ranker",
    value=True,
    help="Apply re-ranking to improve search result relevance (might be slightly slower)"
)

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Number of chunks to retrieve"
)

# Hidden settings - always enabled for better results
calculate_scores = True  # Always calculate relevance scores
calculate_groundedness = True  # Always calculate groundedness
include_human_eval = False  # Don't use human eval by default

# Helper functions
@st.cache_resource
def get_vector_db():
    """Initialize and return the vector database."""
    try:
        # Create the database directory if it doesn't exist
        os.makedirs("db", exist_ok=True)
        
        # Initialize the vector database
        db = VectorDatabase(
            dimension=EMBEDDING_DIM,
            index_type="L2",
            index_path="db/faiss_index",
            metadata_path="db/metadata.pickle"
        )
        
        return db
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        st.error(f"Error initializing vector database: {e}")
        return None

@st.cache_resource
def get_document_processor():
    """Initialize and return the document processor."""
    try:
        # Initialize the document processor with intelligent default settings
        processor = DocumentProcessor(
            embedding_model_path=embedding_model,
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
            extract_tables=True,  # Always extract tables for better context preservation
            extract_images=True   # Always extract images to get complete information
        )
        
        return processor
    except Exception as e:
        logger.error(f"Error initializing document processor: {e}")
        st.error(f"Error initializing document processor: {e}")
        return None

def process_file(uploaded_file, db, processor):
    """Process an uploaded file and add to the database."""
    try:
        if uploaded_file is None:
            return None
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process the file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                start_time = time.time()
                
                # Process and embed the document (always use intelligent chunking)
                result = processor.process_and_embed(tmp_path, chunk_by_title=True)
                
                # Add to vector database
                file_id = result["file_id"]
                chunks = result["chunks"]
                metadata = result["metadata"]
                
                # Prepare data for the database
                embeddings = np.array([chunk["embedding"] for chunk in chunks])
                chunk_texts = [chunk["text"] for chunk in chunks]
                chunk_metadata = [chunk["metadata"] for chunk in chunks]
                
                # Add to the database
                db.add_embeddings(
                    embeddings=embeddings,
                    chunk_contents=chunk_texts,
                    chunk_metadata=chunk_metadata,
                    file_id=file_id,
                    document_type=metadata.get("format", "unknown")
                )
                
                # Save the database
                db.save()
                
                processing_time = time.time() - start_time
                
                return {
                    "file_id": file_id,
                    "file_name": uploaded_file.name,
                    "chunk_count": len(chunks),
                    "processing_time": processing_time,
                    "metadata": metadata
                }
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {e}")
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return None

def search_database(query, db, processor, file_ids=None):
    """Search the database for relevant chunks."""
    try:
        # Generate query embedding
        query_embedding = processor.embedding_model.encode(query)
        
        # Search the database
        results = db.search(
            query_embedding=query_embedding,
            k=top_k,
            file_ids=file_ids
        )
        
        # Apply re-ranking if enabled
        if use_reranker and calculate_scores:
            results = score_results(
                query=query,
                answer="",  # No answer yet
                chunks=results,
                include_human_eval=False
            )["chunks"]
            
            # Sort by relevance score
            results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return results
    except Exception as e:
        logger.error(f"Error searching database: {e}")
        st.error(f"Error searching database: {e}")
        return []

def generate_answer(query, chunks):
    """
    Generate an answer based on retrieved chunks.
    This is a placeholder - in production, you'd call an LLM API here.
    """
    if not chunks:
        return "No relevant information found."
    
    # Simple extractive answer for demonstration
    # In production, replace with an actual LLM call
    combined_text = "\n\n".join([chunk.get("text", "") for chunk in chunks[:3]])
    
    # Create a simple answer (in production, use an LLM for this)
    answer = f"Based on the retrieved information:\n\n{combined_text[:500]}...\n\n(This is a placeholder answer. In production, an LLM would generate a proper response based on the retrieved chunks.)"
    
    return answer

def score_answer(query, answer, chunks):
    """Score the answer for relevance and groundedness."""
    if not calculate_scores:
        return None
    
    try:
        scores = score_results(
            query=query,
            answer=answer,
            chunks=chunks,
            include_human_eval=include_human_eval
        )
        
        return scores
    except Exception as e:
        logger.error(f"Error calculating scores: {e}")
        st.error(f"Error calculating scores: {e}")
        return None

# Session state initialization
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Initialize vector database and document processor
vector_db = get_vector_db()
doc_processor = get_document_processor()

# Main UI
st.title("Intelligent Vector Database RAG System")

# Upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload any document type (PDF, Word, Excel, CSV, Text, HTML, Images, etc.)",
    type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "csv", "txt", "md", "html", "htm", "jpg", "jpeg", "png", "gif", "tiff", "eod"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Check if this file is already processed
        if uploaded_file.name in st.session_state.uploaded_files:
            st.info(f"{uploaded_file.name} already processed.")
            continue
        
        # Process the file
        result = process_file(uploaded_file, vector_db, doc_processor)
        
        if result:
            st.success(f"Successfully processed {uploaded_file.name} into {result['chunk_count']} intelligent chunks in {result['processing_time']:.2f} seconds. The system automatically preserved document structure, tables, and context between chunks.")
            
            # Store in session state
            st.session_state.uploaded_files[uploaded_file.name] = result

# File selection for search
st.header("Select Files for Search")

if st.session_state.uploaded_files:
    # Create checkboxes for each file
    file_names = list(st.session_state.uploaded_files.keys())
    
    # "Select All" option
    select_all = st.checkbox("Select All Files")
    
    if select_all:
        st.session_state.selected_files = file_names
    else:
        # Individual file selection
        selected_files = []
        cols = st.columns(3)
        for i, file_name in enumerate(file_names):
            col_idx = i % 3
            with cols[col_idx]:
                selected = st.checkbox(file_name, value=file_name in st.session_state.selected_files)
                if selected:
                    selected_files.append(file_name)
        
        st.session_state.selected_files = selected_files
    
    if st.session_state.selected_files:
        st.info(f"Selected {len(st.session_state.selected_files)} files for search.")
    else:
        st.warning("No files selected. All files will be searched.")
else:
    st.warning("No files uploaded. Please upload documents first.")

# Search section
st.header("Search Documents")
query = st.text_input("Enter your query")

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        # Get selected file IDs
        if st.session_state.selected_files:
            file_ids = [st.session_state.uploaded_files[name]["file_id"] for name in st.session_state.selected_files]
        else:
            file_ids = None  # Search all files
        
        # Perform search
        with st.spinner("Searching..."):
            start_time = time.time()
            results = search_database(query, vector_db, doc_processor, file_ids)
            search_time = time.time() - start_time
        
        if results:
            st.success(f"Found {len(results)} relevant chunks in {search_time:.2f} seconds.")
            
            # Generate an answer
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, results)
            
            # Display the answer
            st.subheader("Answer")
            st.write(answer)
            
            # Calculate scores if enabled
            scores = None
            if calculate_scores or calculate_groundedness:
                with st.spinner("Calculating scores..."):
                    scores = score_answer(query, answer, results)
            
            # Display scores if available
            if scores:
                st.subheader("Quality Scores")
                score_cols = st.columns(3)
                
                with score_cols[0]:
                    st.metric("Average Relevance", f"{scores['avg_relevance_score']:.2f}")
                
                with score_cols[1]:
                    if "overall_groundedness" in scores:
                        st.metric("Groundedness", f"{scores['overall_groundedness']:.2f}")
                
                with score_cols[2]:
                    if "human_eval" in scores:
                        st.metric("Human-like Rating", f"{scores['human_eval']['overall']:.2f}")
            
            # Display retrieved chunks
            st.subheader("Retrieved Information")
            
            for i, chunk in enumerate(results):
                with st.expander(f"Chunk {i+1} - {chunk.get('metadata', {}).get('filename', 'Unknown')}"):
                    # Display chunk text
                    st.write(chunk.get("text", "No text available"))
                    
                    # Display scores if available
                    if "relevance_score" in chunk:
                        st.progress(min(1.0, chunk["relevance_score"]))
                        st.caption(f"Relevance Score: {chunk['relevance_score']:.2f}")
                    
                    # Display metadata
                    with st.expander("Metadata"):
                        st.json(chunk.get("metadata", {}))
            
            # Add to search history
            st.session_state.search_history.append({
                "query": query,
                "result_count": len(results),
                "answer": answer,
                "scores": scores,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            st.warning("No relevant information found. Try a different query or upload more documents.")

# Search history
if st.session_state.search_history:
    st.header("Search History")
    
    for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
        with st.expander(f"{search['timestamp']} - {search['query']}"):
            st.write(f"Query: {search['query']}")
            st.write(f"Results: {search['result_count']}")
            st.write(f"Answer: {search['answer'][:200]}...")
            
            if search['scores']:
                st.write("Scores:")
                st.json({k: v for k, v in search['scores'].items() if k not in ["chunks", "query"]})

# Database stats
st.header("Vector Database Stats")
stats = vector_db.get_stats()

stats_cols = st.columns(3)
with stats_cols[0]:
    st.metric("Total Chunks", stats.get("total_chunks", 0))

with stats_cols[1]:
    st.metric("Total Files", stats.get("total_files", 0))

with stats_cols[2]:
    st.metric("Embedding Dimension", stats.get("dimension", EMBEDDING_DIM))

# Document types
if stats.get("document_types"):
    st.subheader("Document Types")
    
    # Count document types
    doc_types = {}
    for _, doc_type in stats.get("document_types", {}).items():
        if doc_type in doc_types:
            doc_types[doc_type] += 1
        else:
            doc_types[doc_type] = 1
    
    # Display as a table
    doc_type_df = pd.DataFrame(
        {"Count": doc_types.values()},
        index=doc_types.keys()
    )
    st.bar_chart(doc_type_df)

# Footer
st.markdown("---")
st.caption("Vector Database RAG System - Built with Faiss, Streamlit, and Hugging Face Models")

# Main function
def main():
    pass

if __name__ == "__main__":
    main()
