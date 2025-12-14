## Project Overview

Implemented the **RAG (Retrieval-Augmented Generation)** method using **OLLAMA LLM**.

## Implementation Details

1. Used **FastAPI** to upload the image file in `getting_image_file.py`.

2. Copied the uploaded file path and used the **Poppler** path to convert the image-based PDF for OCR processing.  
   Then used the **Tesseract** library to convert the image into string format.

3. Used **RecursiveCharacterTextSplitter** from **LangChain** to split the extracted text into chunks.

4. Used the **OnnxMiniEmbeddings** class to load the Hugging Face  
   `Xenova/all-MiniLM-L6-v2` model.  
   The `sentence-transformers/all-MiniLM-L6-v2` tokenizer was used to load the model inside a callable input function.  
   Implemented both `embed_query` and `embed_documents` methods to handle single-line and multi-line OCR text.

5. Used the **OnnxMiniEmbeddings** class in `RAG_Request.py` as the embedding method.  
   Integrated **Chroma** as the vector database to store numerical embeddings in vector DB format.

6. Used **Uvicorn** to run the FastAPI server and process RAG requests through API endpoints.

7. Validated the RAG responses using the **Requests** library.
