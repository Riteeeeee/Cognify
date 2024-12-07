import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io
import requests
from io import BytesIO
import json  # Import the json module for parsing
# Load the saved Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load saved embeddings
embeddings = torch.load('embeddings.pt')

# Load descriptions and images
with open('descriptions.pkl', 'rb') as f:
    descriptions = pickle.load(f)

with open('images.pkl', 'rb') as f:
    images = pickle.load(f)


# Function to find top N matches
def find_top_n_matches(prompt, embeddings, descriptions, images, top_n=5):
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(prompt_embedding, embeddings)[0]

    # Get the top N similarities
    top_results = similarities.topk(k=top_n)

    # Retrieve the indices of the top N results
    top_indices = top_results.indices.tolist()
    top_scores = top_results.values.tolist()

    # Get the top N descriptions, images, and their similarity scores
    top_descriptions = [descriptions[idx] for idx in top_indices]
    top_images = [images[idx] for idx in top_indices]

    return top_descriptions, top_images, top_scores


# Streamlit interface
st.title("Product Description Finder")

prompt = st.text_input("Enter your search prompt:")

if prompt:
    top_descriptions, top_images, top_scores = find_top_n_matches(prompt, embeddings, descriptions, images, top_n=5)

    for i in range(len(top_descriptions)):
        st.write(f"**Score:** {top_scores[i]:.4f}")
        st.write(f"**Description:** {top_descriptions[i]}")

        # Parse the string representation of the list into an actual list
        try:
            image_urls_list = json.loads(top_images[i])

            if isinstance(image_urls_list, list):
                # Create columns for image display
                num_columns = 3
                columns = st.columns(num_columns)

                for j, image_url in enumerate(image_urls_list):
                    try:
                        # Check if image_url is valid
                        if isinstance(image_url, str) and image_url.startswith("http"):
                            # Fetch the image from the URL
                            response = requests.get(image_url)
                            response.raise_for_status()  # Check for request errors
                            image = Image.open(BytesIO(response.content))

                            # Determine which column to place the image in
                            col_index = j % num_columns
                            with columns[col_index]:
                                st.image(image, caption=f"Image {i + 1}-{j + 1}")
                        else:
                            st.write(f"Invalid URL: {image_url}")
                    except Exception as e:
                        st.write(f"Error loading image {i + 1}-{j + 1}: {e}")
            else:
                st.write(f"Expected a list of URLs but got: {image_urls_list}")
        except json.JSONDecodeError as e:
            st.write(f"Error parsing image URLs for entry {i + 1}: {e}")

