import os
import re
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------
# 1. CREATE FOLDER STRUCTURE
# --------------------------------

os.makedirs("dataset/documents", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# --------------------------------
# 2. CREATE 10 EDUCATION DOCUMENTS
# --------------------------------

documents = {
    "document1.txt": """Education is important for every student. It helps students develop knowledge, skills, confidence, and good values. Education prepares students for their future careers and helps them become responsible citizens.""",

    "document2.txt": """Online learning allows students to study using computers, mobile phones, and the internet. Students can attend virtual classes, watch recorded lectures, and access digital learning materials from anywhere.""",

    "document3.txt": """Computer science is the study of computers, programming, algorithms, and software development. It helps students understand how computer systems work and how technology solves real-world problems.""",

    "document4.txt": """Artificial Intelligence is a technology that enables machines to perform tasks that usually require human intelligence. AI is used in education, healthcare, transportation, business, and many other fields.""",

    "document5.txt": """Programming is the process of writing instructions for computers. Popular programming languages include Python, Java, C, and JavaScript. Programming helps students develop logical thinking and problem-solving skills.""",

    "document6.txt": """Digital literacy means the ability to use digital devices, software, and the internet effectively. Students need digital literacy to find information, communicate online, and participate in modern education.""",

    "document7.txt": """Distance learning allows students to continue their education without attending a physical classroom. It uses online platforms, video lessons, digital assignments, and virtual communication tools.""",

    "document8.txt": """Practical learning helps students understand concepts through experiments, projects, laboratory activities, and real-world applications. It improves creativity, confidence, and technical skills.""",

    "document9.txt": """Libraries provide students with books, journals, newspapers, and digital resources. A library supports independent learning, research activities, reading habits, and academic development.""",

    "document10.txt": """Technology improves education through smart classrooms, online courses, digital libraries, educational applications, and virtual laboratories. It makes learning more accessible and interactive for students."""
}

documents_path = "dataset/documents"

# Create documents only if the folder is empty
existing_files = [
    f for f in os.listdir(documents_path)
    if f.endswith(".txt")
]

if not existing_files:
    for filename, content in documents.items():
        with open(os.path.join(documents_path, filename), "w", encoding="utf-8") as file:
            file.write(content)

    print("10 education documents created successfully!")

# --------------------------------
# 3. LOAD DOCUMENTS
# --------------------------------

texts = []
ids = []
metadatas = []

for filename in sorted(os.listdir(documents_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(documents_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        if content.strip():
            texts.append(content)
            ids.append(filename.replace(".txt", ""))

            metadatas.append({
                "title": filename.replace(".txt", "").replace("_", " ").title(),
                "category": "Education",
                "source": filename
            })

print("Total documents loaded:", len(texts))

if not texts:
    raise ValueError("No text documents found!")

# --------------------------------
# 4. GENERATE EMBEDDINGS
# --------------------------------

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

np.save("dataset/embeddings.npy", embeddings)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# --------------------------------
# 5. CREATE CHROMADB COLLECTION
# --------------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="education_documents",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents stored in ChromaDB successfully!")

# --------------------------------
# 6. CREATE 10 USER QUERIES
# --------------------------------

queries = [
    "Why is education important?",
    "What are the benefits of online learning?",
    "What is computer science?",
    "Explain artificial intelligence.",
    "Why should students learn programming?",
    "What is digital literacy?",
    "How does distance learning work?",
    "What is practical learning?",
    "How do libraries help students?",
    "How does technology improve education?"
]

pd.DataFrame({
    "query": queries
}).to_csv("dataset/queries.csv", index=False)

# --------------------------------
# 7. SIMILARITY SEARCH AND KEYWORD SEARCH
# --------------------------------

results_data = []

for query in queries:

    print("\n--------------------------------")
    print("USER QUERY:", query)
    print("--------------------------------")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    search_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(5, len(texts)),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_docs = search_results["documents"][0]
    retrieved_ids = search_results["ids"][0]
    retrieved_metadata = search_results["metadatas"][0]
    distances = search_results["distances"][0]

    # Keyword-based search
    query_words = set(re.findall(r"\w+", query.lower()))
    keyword_scores = []

    for i, content in enumerate(texts):
        document_words = set(re.findall(r"\w+", content.lower()))
        score = len(query_words.intersection(document_words))
        keyword_scores.append((i, score))

    keyword_scores.sort(key=lambda x: x[1], reverse=True)

    keyword_ranks = {
        ids[index]: rank
        for rank, (index, score) in enumerate(keyword_scores[:5], start=1)
    }

    # Save Top-5 similarity results
    for rank, (doc_id, content, metadata, distance) in enumerate(
        zip(retrieved_ids, retrieved_docs, retrieved_metadata, distances),
        start=1
    ):
        similarity = 1 - float(distance)

        print("\nRank:", rank)
        print("Document:", metadata["title"])
        print("Similarity Score:", round(similarity, 4))
        print("Content:", content)

        results_data.append({
            "query": query,
            "similarity_rank": rank,
            "document_id": doc_id,
            "document_title": metadata["title"],
            "category": metadata["category"],
            "source": metadata["source"],
            "document_content": content,
            "similarity_score": round(similarity, 4),
            "keyword_rank": keyword_ranks.get(doc_id, "Not in Top 5")
        })

# --------------------------------
# 8. SAVE RETRIEVED RESULTS
# --------------------------------

results_df = pd.DataFrame(results_data)

results_df.to_csv(
    "dataset/retrieved_results.csv",
    index=False
)

print("\nRetrieval results saved successfully!")

# --------------------------------
# 9. DISPLAY COMPARISON
# --------------------------------

print("\nKEYWORD SEARCH AND SIMILARITY SEARCH COMPARISON")

for query in queries:
    query_results = results_df[results_df["query"] == query]

    print("\nQuery:", query)
    print("Similarity Search Top-5:")

    print(
        query_results[
            ["similarity_rank", "document_title", "similarity_score"]
        ].to_string(index=False)
    )

    print("Keyword Search Top-5:")

    keyword_results = query_results[
        query_results["keyword_rank"] != "Not in Top 5"
    ].sort_values("keyword_rank")

    print(
        keyword_results[
            ["keyword_rank", "document_title"]
        ].to_string(index=False)
    )

# --------------------------------
# 10. FINAL OUTPUT
# --------------------------------

print("\n================================")
print("WEEK 11 HOME ACTIVITY COMPLETED")
print("================================")

print("Files created:")
print("1. dataset/queries.csv")
print("2. dataset/retrieved_results.csv")
print("3. dataset/embeddings.npy")
print("4. chroma_db/ database files")

print("\nTotal documents:", len(texts))
print("Total queries:", len(queries))
print("Total retrieved results:", len(results_df))