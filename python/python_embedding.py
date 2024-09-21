import time
from sentence_transformers import SentenceTransformer

def generate_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)

def main():
    sentences = [
        "This is a test sentence.",
        "Let's see how fast we can generate embeddings.",
        "Performance testing is crucial for optimization."
    ]

    start_time = time.time()
    
    for sentence in sentences:
        embedding = generate_embedding(sentence)
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding[:5]}...")  # Print first 5 dimensions for brevity

    end_time = time.time()
    print(f"Total time taken for embedding {len(sentences)} sentences: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
