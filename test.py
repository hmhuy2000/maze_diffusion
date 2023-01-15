
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.train()

#Our sentences we like to encode
prompt = ('Initialize grid','Add a room in the top left with size 3','Initialize grid','Initialize grid')

#Sentences are encoded by calling model.encode()
embeddings = model.encode(prompt)

#Print the embeddings
print(embeddings.shape)
print("Num params: ", sum(p.numel() for p in model.parameters()))