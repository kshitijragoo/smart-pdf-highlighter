# src/category_manager.py

import json
import os
from typing import Dict

default_categories = {
    "Key Concepts": "Sentences that introduce or explain the main ideas or principles.",
    "Examples": "Sentences that provide specific instances or illustrations of the concepts.",
    "Definitions": "Sentences that clarify terms or explain specific aspects related to the topic."
}

CATEGORY_FILE = "categories.json"

def load_categories() -> Dict[str, str]:
    if not os.path.exists(CATEGORY_FILE):
        save_categories(default_categories)  # Initialize with default categories
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_categories(categories: Dict[str, str]):
    with open(CATEGORY_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=4, ensure_ascii=False)

def reset_categories():
    save_categories(default_categories)
