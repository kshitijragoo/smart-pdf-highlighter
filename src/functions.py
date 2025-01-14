# src/functions.py

"""
This module provides functions for generating a highlighted PDF with important sentences
categorized by user-selected criteria.

The main function, `generate_highlighted_pdf`, takes an input PDF file, user-selected
criteria, and their corresponding colors as input. It utilizes a pre-trained sentence
embedding model and ChatGPT-4-mini's API to identify and categorize important sentences,
which are then highlighted in the PDF with the specified colors.

Other utility functions in this module include functions for loading a sentence embedding
model, encoding sentences, computing similarity matrices, building graphs, ranking sentences,
clustering sentence embeddings, and splitting text into sentences.

Note: This module requires the PyMuPDF, networkx, numpy, torch, sentence_transformers,
sklearn, and requests libraries to be installed.
"""

import logging
import os
from typing import BinaryIO, List, Tuple, Dict

import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
PAGERANK_THRESHOLD_RATIO = 0.15
NUM_CLUSTERS_RATIO = 0.05
MIN_WORDS = 10

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_sentence_model(revision: str = None) -> SentenceTransformer:
    """
    Load a pre-trained sentence embedding model.

    Args:
        revision (str): Optional parameter to specify the model revision.

    Returns:
        SentenceTransformer: A pre-trained sentence embedding model.
    """
    return SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)

# Load the model once to improve efficiency
MODEL = load_sentence_model()

def encode_sentence(model: SentenceTransformer, sentence: str) -> torch.Tensor:
    """
    Encode a sentence into a fixed-dimensional vector representation.

    Args:
        model (SentenceTransformer): A pre-trained sentence embedding model.
        sentence (str): Input sentence.

    Returns:
        torch.Tensor: Encoded sentence vector.
    """
    model.eval()  # Set the model to evaluation mode

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Disable gradient tracking
        return model.encode(sentence, convert_to_tensor=True).to(device)

def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute the cosine similarity matrix between sentence embeddings.

    Args:
        embeddings (torch.Tensor): Sentence embeddings.

    Returns:
        np.ndarray: Normalized cosine similarity matrix.
    """
    scores = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    )
    similarity_matrix = scores.cpu().numpy()
    normalized_adjacency_matrix = similarity_matrix / similarity_matrix.sum(
        axis=1, keepdims=True
    )
    return normalized_adjacency_matrix

def build_graph(normalized_adjacency_matrix: np.ndarray) -> nx.DiGraph:
    """
    Build a directed graph from a normalized adjacency matrix.

    Args:
        normalized_adjacency_matrix (np.ndarray): Normalized adjacency matrix.

    Returns:
        nx.DiGraph: Directed graph.
    """
    return nx.DiGraph(normalized_adjacency_matrix)

def rank_sentences(graph: nx.DiGraph, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences based on PageRank scores.

    Args:
        graph (nx.DiGraph): Directed graph.
        sentences (List[str]): List of sentences.

    Returns:
        List[Tuple[str, float]]: Ranked sentences with their PageRank scores.
    """
    pagerank_scores = nx.pagerank(graph)
    ranked_sentences = sorted(
        zip(sentences, pagerank_scores.values()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked_sentences

def cluster_sentences(
    embeddings: torch.Tensor, num_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster sentence embeddings using K-means clustering.

    Args:
        embeddings (torch.Tensor): Sentence embeddings.
        num_clusters (int): Number of clusters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cluster assignments and cluster centers.
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Explicitly set n_init
    cluster_assignments = kmeans.fit_predict(embeddings.cpu())
    cluster_centers = kmeans.cluster_centers_
    return cluster_assignments, cluster_centers

def get_middle_sentence(cluster_indices: np.ndarray, sentences: List[str]) -> List[str]:
    """
    Get the middle sentence from each cluster.

    Args:
        cluster_indices (np.ndarray): Cluster assignments.
        sentences (List[str]): List of sentences.

    Returns:
        List[str]: Middle sentences from each cluster.
    """
    middle_indices = [
        int(np.median(np.where(cluster_indices == i)[0]))
        for i in range(max(cluster_indices) + 1)
    ]
    middle_sentences = [sentences[i] for i in middle_indices]
    return middle_sentences

def split_text_into_sentences(text: str, min_words: int = MIN_WORDS) -> List[str]:
    """
    Split text into sentences.

    Args:
        text (str): Input text.
        min_words (int): Minimum number of words for a valid sentence.

    Returns:
        List[str]: List of sentences.
    """
    sentences = []
    for s in text.split("."):
        s = s.strip()
        # filtering out short sentences and sentences that contain more than 40% digits
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / len(s)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc):
    """Generator to yield text per page from the PDF, for memory efficiency for large PDFs."""
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()

def analyze_text_with_chatgpt(text: str, criteria: List[str], api_key: str) -> Dict[str, List[str]]:
    """
    Sends text to ChatGPT-4-mini API to identify parts based on criteria.

    Args:
        text (str): The text to analyze.
        criteria (List[str]): List of criteria to identify.
        api_key (str): API key for authentication.

    Returns:
        Dict[str, List[str]]: Mapping of criteria to identified sentences.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Analyze the following text and categorize each sentence into the following criteria: {', '.join(criteria)}.\n"
        f"Provide the output in JSON format with each criterion as a key and a list of corresponding sentences as values.\n\n"
        f"Text:\n{text}"
    )

    data = {
        "model": "chatgpt-4-mini",
        "prompt": prompt,
        "max_tokens": 2000,
        "temperature": 0.5,
        "n": 1,
        "stop": None
    }

    try:
        response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # Assuming the API returns a 'choices' list with 'text' field
        chatgpt_output = result['choices'][0]['text'].strip()

        # Parse the JSON output
        categorized_sentences = parse_chatgpt_response(chatgpt_output)
        return categorized_sentences

    except requests.exceptions.RequestException as e:
        logger.error(f"ChatGPT API request failed: {e}")
        return {}
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing ChatGPT API response: {e}")
        return {}

def parse_chatgpt_response(response_text: str) -> Dict[str, List[str]]:
    """
    Parse the JSON response from ChatGPT.

    Args:
        response_text (str): JSON-formatted string from ChatGPT.

    Returns:
        Dict[str, List[str]]: Parsed mapping of criteria to sentences.
    """
    import json
    try:
        categorized = json.loads(response_text)
        # Ensure that all criteria keys exist
        return {key: value for key, value in categorized.items() if isinstance(value, list)}
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from ChatGPT response.")
        return {}

def generate_highlighted_pdf(
    input_pdf_file: BinaryIO,
    criteria: List[str],
    criteria_colors: Dict[str, Tuple[float, float, float]],
    model: SentenceTransformer = MODEL
) -> bytes or str:
    """
    Generate a highlighted PDF with important sentences categorized by criteria.

    Args:
        input_pdf_file: Input PDF file object.
        criteria (List[str]): List of criteria for categorization.
        criteria_colors (Dict[str, Tuple[float, float, float]]): Mapping of criteria to RGB colors.
        model (SentenceTransformer): Pre-trained sentence embedding model.

    Returns:
        bytes: Highlighted PDF content.
        str: Error message if processing fails.
    """
    # Retrieve GPT API key from environment variables
    api_key = os.getenv("GPT_API_KEY")
    if not api_key:
        logger.error("GPT_API_KEY environment variable not set.")
        return "Server configuration error: API key not found."

    try:
        with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
            num_pages = doc.page_count

            if num_pages > MAX_PAGE:
                return f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages."

            sentences = []
            for page_text in extract_text_from_pages(doc):  # Memory efficient
                sentences.extend(split_text_into_sentences(page_text))

            len_sentences = len(sentences)

            if len_sentences > MAX_SENTENCES:
                return f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences."

            # Encode sentences
            embeddings = encode_sentence(model, sentences)

            # Compute similarity matrix and build graph
            similarity_matrix = compute_similarity_matrix(embeddings)
            graph = build_graph(similarity_matrix)

            # Rank sentences using PageRank
            ranked_sentences = rank_sentences(graph, sentences)

            # Determine top sentences based on PageRank threshold
            pagerank_threshold = int(len(ranked_sentences) * PAGERANK_THRESHOLD_RATIO) + 1
            top_pagerank_sentences = [
                sentence[0] for sentence in ranked_sentences[:pagerank_threshold]
            ]

            # Determine number of clusters
            num_clusters = int(len_sentences * NUM_CLUSTERS_RATIO) + 1
            cluster_assignments, _ = cluster_sentences(embeddings, num_clusters)

            # Get middle sentences from each cluster
            center_sentences = get_middle_sentence(cluster_assignments, sentences)

            # Combine important sentences
            important_sentences = list(set(top_pagerank_sentences + center_sentences))

            # Analyze sentences with ChatGPT-4-mini to categorize them
            chatgpt_input_text = " ".join(important_sentences) 
            categorized_sentences = analyze_text_with_chatgpt(chatgpt_input_text, criteria, api_key)

            if not categorized_sentences:
                logger.error("No categorized sentences returned from ChatGPT.")
                return "Error processing the PDF with AI."

            # Iterate through each page to apply highlights
            for i in range(num_pages):
                try:
                    page = doc[i]

                    for category, sentences in categorized_sentences.items():
                        color = criteria_colors.get(category, (1, 1, 0))  # Default to yellow if not found
                        for sentence in sentences:
                            rects = page.search_for(sentence)
                            for rect in rects:
                                annot = page.add_highlight_annot(rect)
                                annot.set_colors(stroke=color)
                                annot.update()
                except Exception as e:
                    logger.error(f"Error processing page {i}: {e}")

            output_pdf = doc.write()

        return output_pdf

    except Exception as e:
        logger.error(f"Error generating highlighted PDF: {e}")
        return "An error occurred while processing the PDF."
