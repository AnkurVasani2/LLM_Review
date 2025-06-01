import os
import warnings
from unstructured.partition.pdf import partition_pdf


def retrieve_from_pdf(file_path, output_dir=None):
    """
    Extracts text, images (as base64 from CompositeElements), and tables from a PDF.

    Args:
        file_path (str): The path to the PDF file.
        output_dir (str, optional): Directory for output (not used, kept for compatibility).

    Returns:
        tuple:
            - element_summary (dict): counts of each element type extracted.
            - chunks (list): the raw list of extracted elements from partition_pdf.
    """
    warnings.filterwarnings("ignore", message="CropBox missing")

    print(f"Retrieving from PDF: {file_path}")

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_images_in_blocks=True,
        extract_images_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    
    element_summary = {}
    for el in chunks:
        el_type = type(el).__name__
        element_summary[el_type] = element_summary.get(el_type, 0) + 1

    return element_summary, chunks


def get_images_base64(chunks):
    """
    (Optional helper) Extracts a list of base64‐encoded images from CompositeElements.
    """
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for sub_el in chunk.metadata.orig_elements:
                    if "Image" in str(type(sub_el)) and hasattr(sub_el.metadata, "image_base64"):
                        images_b64.append(sub_el.metadata.image_base64)
    return images_b64


def separate_elements(chunks):
    """
    Separates elements into text (CompositeElement), tables, and collects base64 images.

    Returns:
        tuple: (list_of_text_elements, list_of_base64_images, list_of_table_elements)
    """
    text_elements = []
    table_elements = []

    for chunk in chunks:
        type_str = str(type(chunk))
        if "Table" in type_str:
            table_elements.append(chunk)
        elif "CompositeElement" in type_str:
            text_elements.append(chunk)

    image_elements = get_images_base64(chunks)
    return text_elements, image_elements, table_elements