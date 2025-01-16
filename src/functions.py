# src/functions.py

import logging
import os
from typing import BinaryIO, List, Tuple, Dict

import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import requests
import json

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000

# Logger configuration

# Ensure logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Adjust to your desired level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "app.log")),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)
logger = logging.getLogger(__name__)

def split_text_into_sentences(text: str, min_words: int = 10) -> List[str]:
    """
    Split text into sentences.
    """
    sentences = []
    for s in text.split("."):
        s = s.strip()
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / len(s)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc):
    """Generator to yield text per page from the PDF."""
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()

def analyze_text_with_chatgpt(sentences_dict: Dict[int, str], criteria: List[str], api_key: str) -> Dict[str, List[int]]:
    """
    Sends a numbered dictionary of sentences to ChatGPT to categorize them based on criteria.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # print(sentences_dict)
    # Convert the sentences_dict to a JSON-formatted string
    sentences_json = json.dumps(sentences_dict, ensure_ascii=False)

    prompt = (
        f"Categorize the following sentences into these criteria: {', '.join(criteria)}. "
        f"Respond ONLY in JSON format, with no additional explanations or comments.\n\n"
        f"Sentences:\n{sentences_json}"
    )


    data = {
        "model": "gpt-4o-mini",  # Ensure this matches your model subscription
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.5
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()

        # Parse the assistant's reply
        chatgpt_output = result['choices'][0]['message']['content'].strip()
        logger.debug(f"ChatGPT Output: {chatgpt_output}")  # Optional: Log the output for debugging

        # Parse the JSON output
        categorized_sentences = parse_chatgpt_response(chatgpt_output)
        return categorized_sentences

    except requests.exceptions.RequestException as e:
        logger.error(f"ChatGPT API request failed: {e}")
        return {}
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing ChatGPT API response: {e}")
        return {}

def parse_chatgpt_response(response_text: str) -> Dict[str, List[int]]:
    """
    Parse the JSON response from ChatGPT to extract criteria and corresponding sentence numbers.
    """
    import json
    try:
        # Strip markdown-style code block delimiters if they exist
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()

        # Decode the JSON content
        categorized = json.loads(response_text)

        # Ensure all keys are valid and values are lists of integers
        return {
            key: value
            for key, value in categorized.items()
            if isinstance(value, list) and all(isinstance(num, int) for num in value)
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from ChatGPT response: {e}")
        return {}


def generate_highlighted_pdf(
    input_pdf_file: BinaryIO,
    criteria: List[str],
    criteria_colors: Dict[str, Tuple[float, float, float]],
    ) -> bytes or str:
    """
    Generate a highlighted PDF with important sentences categorized by criteria.
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

            print(f"Sentences: {sentences}")
            # Create a numbered dictionary of all sentences
            numbered_sentences = {idx + 1: sentence for idx, sentence in enumerate(sentences)}
            print(f"Numbered Sentences: {numbered_sentences}")

            # Analyze all sentences with ChatGPT-4-mini to categorize them
            categorized_sentences = analyze_text_with_chatgpt(numbered_sentences, criteria, api_key)

            if not categorized_sentences:
                logger.error("No categorized sentences returned from ChatGPT.")
                return "Error processing the PDF with AI."

            # Iterate through each page to apply highlights
            for i in range(num_pages):
                try:
                    page = doc[i]

                    for category, sentence_numbers in categorized_sentences.items():
                        color = criteria_colors.get(category, (1, 1, 0))  # Default to yellow if not found
                        for sentence_num in sentence_numbers:
                            sentence = numbered_sentences.get(sentence_num)
                            if not sentence:
                                logger.warning(f"Sentence number {sentence_num} not found.")
                                continue
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
