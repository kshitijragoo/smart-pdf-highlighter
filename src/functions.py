# src/functions.py

import logging
import os
import re
from typing import BinaryIO, List, Tuple, Dict

import fitz  # PyMuPDF
import requests
import json

from .category_manager import load_categories, reset_categories  # Import category manager

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
CATEGORY_FILE = "categories.json"

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

    Args:
        text (str): The text to split.
        min_words (int): Minimum number of words to consider a valid sentence.

    Returns:
        List[str]: A list of cleaned sentences.
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


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing non-printable characters and normalizing spaces.

    Args:
        text (str): Raw text extracted from PDF.

    Returns:
        str: Cleaned text.
    """
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Keeps only printable ASCII characters
    # Normalize multiple spaces and newlines to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pages(doc) -> List[str]:
    """
    Generator to yield cleaned text per page from the PDF.

    Args:
        doc: PyMuPDF Document object.

    Yields:
        str: Cleaned text of each page.
    """
    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text("text")  # 'text' mode for better extraction
        cleaned_text = clean_text(raw_text)
        yield cleaned_text


def analyze_text_with_chatgpt(sentences_dict: Dict[int, str], criteria: Dict[str, str], api_key: str) -> Dict[str, List[int]]:
    """
    Sends a numbered dictionary of sentences to ChatGPT to categorize them based on criteria.

    Args:
        sentences_dict (Dict[int, str]): Dictionary mapping sentence numbers to sentences.
        criteria (Dict[str, str]): Dictionary mapping category names to their detailed descriptions.
        api_key (str): API key for authentication.

    Returns:
        Dict[str, List[int]]: Mapping of criteria to lists of sentence numbers.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Convert the sentences_dict to a JSON-formatted string
    sentences_json = json.dumps(sentences_dict, ensure_ascii=False)

    # Construct category details string
    category_details = "\n".join([f"- **{key}**: {value}" for key, value in criteria.items()])

    # Optional: Include few-shot examples for better guidance
    # Adjust or remove examples as per your domain
    examples = """
    Examples:
    - Sentence 1: "Support Vector Machines are a type of supervised learning algorithm." -> Key Concepts
    - Sentence 2: "An SVM can classify data points using a hyperplane." -> Key Concepts
    - Sentence 3: "For instance, consider a dataset with two classes." -> Examples
    - Sentence 4: "A support vector machine (SVM) is used for classification tasks." -> Definitions
    """

    prompt = (
        f"Analyze the following sentences and categorize each sentence number into the following criteria based on their descriptions below:\n\n"
        f"**Criteria Definitions:**\n{category_details}\n\n"
        f"**Instructions:**\n"
        f"- Categorize each sentence into one or more of the above criteria.\n"
        f"- Provide the output in JSON format with each criterion as a key and a list of corresponding sentence numbers as values.\n"
        f"- Respond ONLY in JSON format without any additional explanations or comments.\n\n"
        f"{examples}\n\n"
        f"**Sentences:**\n{sentences_json}"
    )

    data = {
        "model": "gpt-4",  # Use the appropriate model name
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
        logger.debug(f"ChatGPT Output: {chatgpt_output}")  # Log the output for debugging

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
    Extract and parse JSON from ChatGPT's response.

    Args:
        response_text (str): The raw response text from ChatGPT.

    Returns:
        Dict[str, List[int]]: Parsed JSON mapping categories to sentence numbers.
    """
    try:
        # Extract JSON using regex to find the content between the JSON block
        json_match = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Attempt to parse the entire response if no code block is found
            json_text = response_text.strip()

        # Parse the JSON content
        categorized = json.loads(json_text)

        # Ensure all keys are valid and values are lists of integers
        return {
            key: value
            for key, value in categorized.items()
            if isinstance(key, str) and isinstance(value, list) and all(isinstance(num, int) for num in value)
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from ChatGPT response: {e}")
        logger.debug(f"Response Text: {response_text}")  # Log the raw response for debugging
        return {}


def generate_highlighted_pdf(
    input_pdf_file: BinaryIO,
    criteria: Dict[str, str],  # Changed from List[str] to Dict[str, str]
    criteria_colors: Dict[str, Tuple[float, float, float]],
) -> bytes or str:
    """
    Generate a highlighted PDF with important sentences categorized by criteria.

    Args:
        input_pdf_file (BinaryIO): The uploaded PDF file.
        criteria (Dict[str, str]): Categories with their detailed descriptions.
        criteria_colors (Dict[str, Tuple[float, float, float]]): Mapping of categories to highlight colors.

    Returns:
        bytes or str: The highlighted PDF as bytes or an error message string.
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

            logger.debug(f"Sentences: {sentences}")
            # Create a numbered dictionary of all sentences
            numbered_sentences = {idx + 1: sentence for idx, sentence in enumerate(sentences)}
            logger.debug(f"Numbered Sentences: {numbered_sentences}")

            # Analyze all sentences with ChatGPT to categorize them
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
