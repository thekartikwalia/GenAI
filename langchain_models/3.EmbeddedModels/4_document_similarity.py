from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Japrit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Values passed in cosine similarity should be 2D-Lists
# result is 2D-List we need 1D
similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
# print(similarity_scores)

# Attach index with each similarity score
idx_attached = list(enumerate(similarity_scores))
# print(idx_attached)

# Sort based on 2nd argument 
sorted_similarity_scores = sorted(idx_attached, key=lambda x:x[1])
# print(sorted_similarity_scores)

# Get highest similarity score pair (index, score)
index, score = highest_similarity_pair = sorted_similarity_scores[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

# Calcutaing document embeddings again n again isn't a good way (it's a costly operation)
# Instead we use VECTOR DATABASES to store these document embeddings 

# This process of getting similarity score, is known as RETRIEVAL process