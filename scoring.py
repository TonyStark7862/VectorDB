import os
import re
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scoring.log")
    ]
)
logger = logging.getLogger("Scoring")

class RelevanceScorer:
    """
    Scores the relevance of retrieved chunks to the query.
    """
    
    def __init__(self, 
                model_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                use_semantic_similarity: bool = True,
                embedding_model_path: Optional[str] = None):
        """
        Initialize the relevance scorer.
        
        Args:
            model_path: Path to the cross-encoder model
            use_semantic_similarity: Whether to also use semantic similarity scoring
            embedding_model_path: Path to embedding model for semantic similarity
        """
        self.model_path = model_path
        self.use_semantic_similarity = use_semantic_similarity
        self.embedding_model_path = embedding_model_path
        self._load_models()
    
    def _load_models(self):
        """Load the scoring models."""
        try:
            # Load cross-encoder model for relevance scoring
            if os.path.exists(self.model_path):
                self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # Fall back to HF model
                logger.warning(f"Model not found at {self.model_path}, loading from Hugging Face")
                self.cross_encoder = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            if torch.cuda.is_available():
                self.cross_encoder = self.cross_encoder.to("cuda")
            
            # Load embedding model for semantic similarity if requested
            if self.use_semantic_similarity:
                if self.embedding_model_path and os.path.exists(self.embedding_model_path):
                    self.embedding_model = SentenceTransformer(self.embedding_model_path)
                else:
                    # Fall back to a default model
                    logger.warning(f"Embedding model not specified or not found, using default model")
                    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            logger.info("Relevance scoring models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading relevance models: {e}")
            raise
    
    def score(self, query: str, chunks: List[Dict], 
             use_cross_encoder: bool = True,
             cross_encoder_weight: float = 0.7) -> List[Dict]:
        """
        Score the relevance of chunks to the query.
        
        Args:
            query: User query
            chunks: List of chunks with text and embeddings
            use_cross_encoder: Whether to use cross-encoder scoring
            cross_encoder_weight: Weight for cross-encoder vs. semantic similarity
            
        Returns:
            List of chunks with relevance scores added
        """
        try:
            if not chunks:
                return []
            
            # Apply cross-encoder scoring
            if use_cross_encoder:
                cross_encoder_scores = self._cross_encoder_score(query, chunks)
            else:
                # Use dummy scores if not using cross-encoder
                cross_encoder_scores = [0.5] * len(chunks)
            
            # Apply semantic similarity scoring
            if self.use_semantic_similarity:
                semantic_scores = self._semantic_similarity_score(query, chunks)
            else:
                # Use dummy scores if not using semantic similarity
                semantic_scores = [0.5] * len(chunks)
            
            # Combine scores
            for i, chunk in enumerate(chunks):
                # Weighted average of cross-encoder and semantic scores
                chunk["relevance_score"] = cross_encoder_scores[i] * cross_encoder_weight + \
                                         semantic_scores[i] * (1 - cross_encoder_weight)
                
                # Store individual scores for analysis
                chunk["cross_encoder_score"] = cross_encoder_scores[i]
                chunk["semantic_similarity_score"] = semantic_scores[i]
            
            # Sort by relevance score
            chunks = sorted(chunks, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            logger.info(f"Scored {len(chunks)} chunks for relevance to query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Error scoring relevance: {e}")
            # Return the original chunks without scores
            for chunk in chunks:
                chunk["relevance_score_error"] = str(e)
            return chunks
    
    def _cross_encoder_score(self, query: str, chunks: List[Dict]) -> List[float]:
        """Score using cross-encoder model."""
        try:
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            
            # Create query-document pairs
            pairs = [[query, text] for text in chunk_texts]
            
            # Score in batches to avoid memory issues
            batch_size = 8
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize and prepare inputs
                inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, 
                                        max_length=512, return_tensors="pt")
                
                if torch.cuda.is_available():
                    inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
                
                # Get scores
                with torch.no_grad():
                    outputs = self.cross_encoder(**inputs)
                    scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                
                all_scores.extend(scores.tolist())
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error in cross-encoder scoring: {e}")
            return [0.5] * len(chunks)  # Return neutral scores on error
    
    def _semantic_similarity_score(self, query: str, chunks: List[Dict]) -> List[float]:
        """Score using semantic similarity."""
        try:
            # Check if chunks already have embeddings
            if all("embedding" in chunk for chunk in chunks):
                chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
            else:
                # Generate embeddings if not already present
                chunk_texts = [chunk.get("text", "") for chunk in chunks]
                chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate cosine similarities
            similarities = util.cos_sim(
                query_embedding, 
                chunk_embeddings
            )[0].cpu().numpy()
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error in semantic similarity scoring: {e}")
            return [0.5] * len(chunks)  # Return neutral scores on error


class GroundednessScorer:
    """
    Scores the groundedness of a generated answer given source chunks.
    """
    
    def __init__(self, model_path: str = "google/t5-small-ssm-nq"):
        """
        Initialize the groundedness scorer.
        
        Args:
            model_path: Path to the groundedness model (NLI or similar)
        """
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the groundedness model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            if os.path.exists(self.model_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # Fall back to HF model for NLI
                logger.warning(f"Model not found at {self.model_path}, loading from Hugging Face")
                self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            logger.info("Groundedness model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading groundedness model: {e}")
            raise
    
    def score(self, 
             answer: str, 
             chunks: List[Dict],
             detailed: bool = False) -> Dict:
        """
        Score the groundedness of an answer given source chunks.
        
        Args:
            answer: Generated answer
            chunks: List of source chunks
            detailed: Whether to return detailed per-sentence scores
            
        Returns:
            Dict with overall and optionally detailed groundedness scores
        """
        try:
            if not chunks or not answer:
                return {"overall_groundedness": 0.0}
            
            # Break answer into sentences
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(answer)
            
            # Combine all chunks into a single context
            context = " ".join([chunk.get("text", "") for chunk in chunks])
            
            # Score each sentence against the context
            sentence_scores = []
            ungrounded_sentences = []
            grounded_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Skip very short sentences that are likely not factual claims
                if len(sentence.split()) < 3:
                    continue
                
                score = self._score_sentence_groundedness(sentence, context)
                sentence_scores.append({"sentence": sentence, "score": score})
                
                if score < 0.5:
                    ungrounded_sentences.append(sentence)
                else:
                    grounded_sentences.append(sentence)
            
            # Calculate overall score
            if sentence_scores:
                overall_score = sum(item["score"] for item in sentence_scores) / len(sentence_scores)
            else:
                overall_score = 0.0
            
            result = {
                "overall_groundedness": overall_score,
                "grounded_sentence_count": len(grounded_sentences),
                "ungrounded_sentence_count": len(ungrounded_sentences)
            }
            
            if detailed:
                result["sentence_scores"] = sentence_scores
                result["ungrounded_sentences"] = ungrounded_sentences
            
            logger.info(f"Scored answer groundedness: {overall_score:.2f} ({len(grounded_sentences)} grounded, {len(ungrounded_sentences)} ungrounded)")
            return result
            
        except Exception as e:
            logger.error(f"Error scoring groundedness: {e}")
            return {"overall_groundedness": 0.0, "error": str(e)}
    
    def _score_sentence_groundedness(self, sentence: str, context: str) -> float:
        """Score the groundedness of a single sentence against the context."""
        try:
            # Truncate context if it's too long
            max_context_length = 512 - len(sentence) - 5  # Account for special tokens
            if len(context) > max_context_length:
                # Try to truncate at sentence boundaries
                contexts = sent_tokenize(context)
                truncated_context = ""
                for ctx in contexts:
                    if len(truncated_context) + len(ctx) <= max_context_length:
                        truncated_context += " " + ctx
                    else:
                        break
                context = truncated_context.strip()
            
            # Prepare inputs for model
            if "t5" in self.model_path.lower():
                # For T5 models
                inputs = self.tokenizer(
                    f"premise: {context} hypothesis: {sentence}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
            else:
                # For RoBERTa and similar models
                inputs = self.tokenizer(
                    context,
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
            
            if torch.cuda.is_available():
                inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            
            # For NLI models: [contradiction, neutral, entailment]
            # We want the entailment score
            groundedness_score = scores[2] if len(scores) == 3 else scores[1]
            
            return float(groundedness_score)
            
        except Exception as e:
            logger.error(f"Error in sentence groundedness scoring: {e}")
            return 0.5  # Return neutral score on error


class HumanEvalSimulator:
    """
    Simulates human evaluation for testing and calibration.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the human evaluation simulator.
        
        Args:
            model_path: Optional path to a model for simulation
        """
        self.model_path = model_path
        self.use_model = model_path is not None
        
        if self.use_model:
            self._load_model()
    
    def _load_model(self):
        """Load a model for simulation."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            if os.path.exists(self.model_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # Fall back to a default model
                logger.warning(f"Model not found at {self.model_path}, loading from Hugging Face")
                self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            logger.info("Human evaluation simulator model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading human eval simulator model: {e}")
            self.use_model = False
    
    def score(self, 
             query: str, 
             answer: str, 
             chunks: List[Dict],
             score_types: List[str] = ["relevance", "correctness", "helpfulness"]) -> Dict:
        """
        Generate simulated human evaluation scores.
        
        Args:
            query: User query
            answer: Generated answer
            chunks: List of retrieved chunks
            score_types: Types of scores to generate
            
        Returns:
            Dict with simulated human eval scores
        """
        results = {}
        
        if self.use_model:
            # Use model to generate scores
            for score_type in score_types:
                # Create a prompt for the specific score type
                prompt = f"Query: {query}\nAnswer: {answer}\nRate the {score_type} on a scale of 1-5:"
                
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    if torch.cuda.is_available():
                        inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Map the output to a 1-5 scale
                        logits = outputs.logits[0].cpu().numpy()
                        score = (np.argmax(logits) / (len(logits) - 1)) * 4 + 1  # Map to 1-5
                        results[score_type] = float(score)
                except Exception as e:
                    logger.error(f"Error in model-based human eval simulation for {score_type}: {e}")
                    results[score_type] = 3.0  # Neutral score on error
        else:
            # Simple heuristic-based simulation
            for score_type in score_types:
                if score_type == "relevance":
                    # Check query term overlap with answer
                    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
                    answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
                    overlap = len(query_terms.intersection(answer_terms)) / max(1, len(query_terms))
                    score = 1 + overlap * 4  # Scale to 1-5
                    
                elif score_type == "correctness":
                    # Check answer term overlap with chunks
                    chunk_text = " ".join([chunk.get("text", "") for chunk in chunks])
                    chunk_terms = set(re.findall(r'\b\w+\b', chunk_text.lower()))
                    answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
                    if not chunk_terms:
                        score = 3.0  # Neutral if no chunks
                    else:
                        overlap = len(answer_terms.intersection(chunk_terms)) / max(1, len(answer_terms))
                        score = 1 + overlap * 4  # Scale to 1-5
                
                elif score_type == "helpfulness":
                    # Simulate based on answer length and structure
                    words = len(re.findall(r'\b\w+\b', answer))
                    paragraphs = len(answer.split('\n\n'))
                    
                    # Longer answers with structure tend to be more helpful (up to a point)
                    length_score = min(1.0, words / 300)  # Caps at 300 words
                    structure_score = min(1.0, paragraphs / 3)  # Caps at 3 paragraphs
                    
                    score = 1 + (length_score * 0.6 + structure_score * 0.4) * 4  # Scale to 1-5
                
                else:
                    # Default neutral score for unknown types
                    score = 3.0
                
                results[score_type] = float(score)
        
        # Add overall score (average of all score types)
        results["overall"] = sum(results.values()) / len(results)
        
        logger.info(f"Generated simulated human eval scores: {results}")
        return results


# Utility function to combine all scoring metrics
def score_results(query: str, 
                 answer: str, 
                 chunks: List[Dict],
                 relevance_model_path: str = None,
                 groundedness_model_path: str = None,
                 include_human_eval: bool = False,
                 human_eval_model_path: str = None) -> Dict:
    """
    Score query, answer, and retrieved chunks on multiple metrics.
    
    Args:
        query: User query
        answer: Generated answer
        chunks: Retrieved chunks
        relevance_model_path: Path to relevance model
        groundedness_model_path: Path to groundedness model
        include_human_eval: Whether to include simulated human eval
        human_eval_model_path: Path to model for human eval simulation
        
    Returns:
        Dict with all scoring metrics
    """
    results = {
        "query": query,
        "answer_length": len(answer),
        "chunk_count": len(chunks)
    }
    
    # Score relevance
    relevance_scorer = RelevanceScorer(
        model_path=relevance_model_path if relevance_model_path else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    scored_chunks = relevance_scorer.score(query, chunks)
    
    results["chunks"] = scored_chunks
    
    if scored_chunks:
        results["top_relevance_score"] = scored_chunks[0].get("relevance_score", 0)
        results["avg_relevance_score"] = sum(c.get("relevance_score", 0) for c in scored_chunks) / len(scored_chunks)
    else:
        results["top_relevance_score"] = 0
        results["avg_relevance_score"] = 0
    
    # Score groundedness
    groundedness_scorer = GroundednessScorer(
        model_path=groundedness_model_path if groundedness_model_path else "roberta-large-mnli"
    )
    groundedness_results = groundedness_scorer.score(answer, chunks, detailed=True)
    results.update(groundedness_results)
    
    # Add human eval if requested
    if include_human_eval:
        human_eval = HumanEvalSimulator(model_path=human_eval_model_path)
        human_scores = human_eval.score(query, answer, chunks)
        results["human_eval"] = human_scores
    
    logger.info(f"Generated comprehensive scores for query: {query[:50]}...")
    return results
