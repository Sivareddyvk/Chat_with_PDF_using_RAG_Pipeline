import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy

# Load SpaCy model for sentence splitting
nlp = spacy.load("en_core_web_sm")

# Step 1: Extract Text from PDF
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of text content for each page in the PDF.
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

# Step 2: Chunking
def chunk_text(text, chunk_size=100):
    """
    Chunk the text into smaller parts for better granularity.

    Args:
        text (list): List of strings (text from PDF pages).
        chunk_size (int): Maximum number of characters per chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    for page in text:
        doc = nlp(page)
        sentences = [sent.text.strip() for sent in doc.sents]
        temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) <= chunk_size:
                temp_chunk += " " + sentence
            else:
                chunks.append(temp_chunk.strip())
                temp_chunk = sentence
        if temp_chunk:
            chunks.append(temp_chunk.strip())
    return chunks

# Step 3: Embedding
def embed_text(chunks):
    """
    Embed the text chunks using a pre-trained SentenceTransformer model.

    Args:
        chunks (list): List of text chunks.

    Returns:
        np.ndarray: Embeddings for the text chunks.
    """
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, convert_to_numpy=True)
        return embeddings, model
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None, None

# Step 4: Store in Vector Database
def store_in_vector_db(embeddings, chunks):
    """
    Store embeddings in a FAISS vector database.

    Args:
        embeddings (np.ndarray): Embeddings to store.
        chunks (list): Corresponding text chunks.

    Returns:
        faiss.IndexFlatL2, list: FAISS index and the stored text chunks.
    """
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, chunks
    except Exception as e:
        print(f"Error storing embeddings in vector database: {e}")
        return None, None

# Step 5: Query Handling
def query_vector_db(query, model, index, chunks, top_k=5):
    """
    Query the FAISS vector database to retrieve relevant text chunks.

    Args:
        query (str): User query.
        model (SentenceTransformer): Pre-trained model for embedding the query.
        index (faiss.IndexFlatL2): FAISS index containing embeddings.
        chunks (list): Corresponding text chunks stored in the database.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Relevant text chunks.
    """
    try:
        query_vector = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_vector, top_k)
        results = [chunks[i] for i in indices[0]]
        return results
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []

# Step 6: Tabular Response Generation
def generate_tabular_response(relevant_chunks):
    """
    Generate a tabular response from relevant chunks if structured data is detected.

    Args:
        relevant_chunks (list): Relevant text chunks retrieved from the database.

    Returns:
        str: Tabular representation or plain text response.
    """
    try:
        structured_data = []
        for chunk in relevant_chunks:
            if "|" in chunk:  # Assuming tabular data uses '|' as a delimiter
                rows = chunk.split("\n")
                structured_data.extend([row.split("|") for row in rows if row])

        if structured_data:
            import pandas as pd
            df = pd.DataFrame(structured_data[1:], columns=structured_data[0])
            return df.to_string(index=False)
        else:
            return "\n".join(relevant_chunks)
    except Exception as e:
        print(f"Error generating tabular response: {e}")
        return "\n".join(relevant_chunks)

# Example Usage
if __name__ == "__main__":
    # Step 1: Extract text from PDF
    pdf_path = "C://Users//muni yaswanth//Downloads//Data-Analytics_All UNITS.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)

    if pdf_text:
        # Step 2: Chunk the extracted text
        chunks = chunk_text(pdf_text)

        # Step 3: Embed the text chunks
        embeddings, model = embed_text(chunks)

        if embeddings is not None:
            # Step 4: Store embeddings in the vector database
            index, stored_chunks = store_in_vector_db(embeddings, chunks)

            if index is not None:
                # Step 5: Query the vector database
                query = "What is the unemployment rate for Bachelor's degrees?"
                relevant_chunks = query_vector_db(query, model, index, stored_chunks)

                # Step 6: Generate response
                response = generate_tabular_response(relevant_chunks)
                print("Response:")
                print(response)
            else:
                print("Failed to create vector database.")
        else:
            print("Failed to generate embeddings.")
    else:
        print("No text extracted from the PDF.")
