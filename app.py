# app.py

"""
Smart PDF Highlighter
This script provides a Streamlit web application for automatically identifying and
highlighting important content within PDF files. It utilizes AI techniques such as
deep learning, clustering, and advanced algorithms such as PageRank to analyze text
and intelligently select key sentences for highlighting. Additionally, it integrates
ChatGPT-4-mini's API to allow multi-color highlights based on user-selected criteria.

Author: Farzad Salajegheh
Date: 2025
"""

import logging
import time
import os

import streamlit as st

from src import generate_highlighted_pdf

# Set environment variable for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the PDF Highlighter tool."""
    try:
        st.set_page_config(page_title="Smart PDF Highlighter", page_icon="./photos/icon.png")
    except Exception as e:
        logger.error(f"Error setting page config: {e}")
        st.error("An unexpected error occurred while setting up the page.")
        return

    st.title("Smart PDF Highlighter")
    show_description()

    # Sidebar for Highlighting Options
    st.sidebar.header("Highlighting Options")
    criteria = st.sidebar.multiselect(
        "Select criteria to highlight:",
        options=["Key Concepts", "Definitions", "Examples", "Important Notes"],
        default=["Key Concepts"]
    )

    # Assign default colors or allow customization
    color_mapping = {}
    for criterion in criteria:
        color = st.sidebar.color_picker(f"Select color for {criterion}", "#FFFF00")
        # Convert hex to RGB tuple normalized between 0 and 1
        color_rgb = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
        color_mapping[criterion] = color_rgb

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.write("PDF file successfully uploaded.")
        process_pdf(uploaded_file, criteria=criteria, colors=color_mapping)

def show_description():
    """Display description of functionality and maximum limits."""
    st.write("""Welcome to **Smart PDF Highlighter**! This tool automatically identifies
        and highlights important content within your PDF files. It utilizes various
        AI techniques such as deep learning and advanced algorithms to analyze the text
        and intelligently select key sentences for highlighting.""")
    st.write("**Maximum Limits:** 40 pages, 2000 sentences.")

def process_pdf(uploaded_file, criteria, colors):
    """Process the uploaded PDF file and generate highlighted PDF."""
    st.write("Generating highlighted PDF...")
    start_time = time.time()

    with st.spinner("Processing..."):
        try:
            result = generate_highlighted_pdf(
                uploaded_file,
                criteria=criteria,
                criteria_colors=colors
            )
        except Exception as e:
            logger.error(f"Unexpected error during PDF generation: {e}")
            st.error("An unexpected error occurred while generating the highlighted PDF.")
            return

        if isinstance(result, str):
            st.error(result)
            logger.error(f"Error generating highlighted PDF: {result}")
            return
        else:
            file = result

    end_time = time.time()
    execution_time = end_time - start_time
    st.success(
        f"Highlighted PDF generated successfully in {execution_time:.2f} seconds."
    )

    st.write("Download the highlighted PDF:")
    st.download_button(
        label="Download",
        data=file,
        file_name="highlighted_pdf.pdf",
    )

if __name__ == "__main__":
    main()
