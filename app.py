# app.py

"""
Smart PDF Highlighter
This script provides a Streamlit web application for automatically identifying and
highlighting important content within PDF files. It utilizes AI techniques such as
deep learning, clustering, and advanced algorithms such as PageRank to analyze text
and intelligently select key sentences for highlighting. Additionally, it integrates
ChatGPT-4's API to allow multi-color highlights based on user-selected criteria.

Author: Farzad Salajegheh
Date: 2025
"""

import logging
import time
import os

import streamlit as st

from src import generate_highlighted_pdf
from src.category_manager import load_categories, save_categories, reset_categories

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

    # Sidebar for Category Management
    st.sidebar.header("Category Management")

    # Load current categories
    categories = load_categories()

    # Display current categories
    st.sidebar.subheader("Current Categories")
    for category, description in categories.items():
        st.sidebar.markdown(f"**{category}**: {description}")

    # Add New Category
    st.sidebar.subheader("Add New Category")
    with st.sidebar.form("add_category_form"):
        new_category_name = st.text_input("Category Name")
        new_category_desc = st.text_area("Category Description")
        submitted_add = st.form_submit_button("Add Category")
        if submitted_add:
            if new_category_name and new_category_desc:
                if new_category_name in categories:
                    st.sidebar.error("Category already exists.")
                else:
                    categories[new_category_name] = new_category_desc
                    save_categories(categories)
                    st.sidebar.success(f"Category '{new_category_name}' added.")
            else:
                st.sidebar.error("Please provide both name and description.")

    # Edit Existing Category
    st.sidebar.subheader("Edit Category")
    with st.sidebar.form("edit_category_form"):
        edit_category = st.selectbox("Select Category to Edit", options=list(categories.keys()))
        if edit_category:
            edited_desc = st.text_area("New Description", value=categories[edit_category])
            submitted_edit = st.form_submit_button("Update Category")
            if submitted_edit:
                if edited_desc:
                    categories[edit_category] = edited_desc
                    save_categories(categories)
                    st.sidebar.success(f"Category '{edit_category}' updated.")
                else:
                    st.sidebar.error("Description cannot be empty.")

    # Delete Category
    st.sidebar.subheader("Delete Category")
    with st.sidebar.form("delete_category_form"):
        delete_category = st.selectbox("Select Category to Delete", options=list(categories.keys()))
        submitted_delete = st.form_submit_button("Delete Category")
        if submitted_delete:
            if delete_category:
                del categories[delete_category]
                save_categories(categories)
                st.sidebar.success(f"Category '{delete_category}' deleted.")
            else:
                st.sidebar.error("Please select a category to delete.")

    # Reset Categories
    st.sidebar.subheader("Reset Categories")
    if st.sidebar.button("Reset to Default"):
        reset_categories()
        categories = load_categories()
        st.sidebar.success("Categories have been reset to default.")

    # Sidebar for Highlighting Options
    st.sidebar.header("Highlighting Options")
    selected_categories = st.sidebar.multiselect(
        "Select categories to highlight:",
        options=list(categories.keys()),
        default=list(categories.keys())
    )

    # Assign colors to selected categories
    criteria_colors = {}
    for category in selected_categories:
        default_color = "#FFFF00"  # Default yellow
        color = st.sidebar.color_picker(f"Select color for {category}", default_color)
        # Convert hex to RGB tuple normalized between 0 and 1
        try:
            color_rgb = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
            criteria_colors[category] = color_rgb
        except:
            st.sidebar.error(f"Invalid color format for {category}. Using default color.")
            criteria_colors[category] = (1, 1, 0)  # Default to yellow

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.write("PDF file successfully uploaded.")
        process_pdf(uploaded_file, criteria=categories, colors=criteria_colors)

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
                input_pdf_file=uploaded_file,
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
