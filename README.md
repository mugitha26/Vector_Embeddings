# Week 11 – Vector Embeddings using Sentence Transformers and ChromaDB

## Project Overview

This project focuses on understanding vector embeddings and semantic search using Sentence Transformers and ChromaDB. It converts text documents into numerical vectors called embeddings and retrieves relevant documents based on user queries.

## Objectives

- Understand vector embeddings.
- Generate embeddings using Sentence Transformers.
- Store and manage embeddings using ChromaDB.
- Perform similarity search.
- Retrieve the top 5 relevant documents for each query.
- Compare keyword-based search with similarity search.

## Technologies Used

- Python
- Sentence Transformers
- ChromaDB
- Pandas
- NumPy
- Jupyter Notebook

## Model Used

`all-MiniLM-L6-v2`

## Project Structure

```text
Vector_Embeddings/
│
├── dataset/
│   ├── documents/
│   │   ├── document1.txt
│   │   ├── document2.txt
│   │   ├── document3.txt
│   │   ├── document4.txt
│   │   ├── document5.txt
│   │   ├── document6.txt
│   │   ├── document7.txt
│   │   ├── document8.txt
│   │   ├── document9.txt
│   │   └── document10.txt
│   │
│   ├── queries.csv
│   ├── retrieved_results.csv
│   └── embeddings.npy
│
├── chroma_db/
│
├── notebook/
│   └── Week11_Vector_Embeddings.ipynb
│
├── vector_store.py
├── README.md
└── requirements.txt
```

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Execution

Run the Python file using:

```bash
python vector_store.py
```

The program performs the following operations:

1. Loads the text documents.
2. Generates embeddings using Sentence Transformers.
3. Saves embeddings in NumPy format.
4. Creates a ChromaDB collection.
5. Stores documents, embeddings, and metadata.
6. Performs similarity search for user queries.
7. Retrieves the top 5 relevant documents.
8. Performs keyword-based search.
9. Compares keyword search and similarity search.
10. Saves the retrieved results into a CSV file.

## Dataset Description

The project uses 10 text documents related to education and technology. It also uses 10 user queries to perform document retrieval.

## Output Files

- `queries.csv` – Stores user queries.
- `retrieved_results.csv` – Stores retrieved documents and search results.
- `embeddings.npy` – Stores generated vector embeddings.
- `chroma_db/` – Stores the ChromaDB vector database.

## Conclusion

This project demonstrates how vector embeddings can be generated using Sentence Transformers and stored in ChromaDB. Similarity search helps retrieve relevant documents based on the meaning of user queries. The comparison between keyword-based search and semantic search provides an understanding of how vector databases improve information retrieval.