from sentence_transformers import SentenceTransformer, util

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# The actual review (reference)
review = "These jeans fit very well and are tight in all the right places and give just enough to not rub anywhere or ever be uncomfortable. Need a pair of nice white jeans for an event? These are priced right and fit very well."

# The generated summary (hypothesis)
summary = "Wear em In Comfort"

# Encode the review and the summary
review_embedding = model.encode(review, convert_to_tensor=True)
summary_embedding = model.encode(summary, convert_to_tensor=True)

# Compute the cosine similarity between the review and the summary
cosine_similarity = util.pytorch_cos_sim(review_embedding, summary_embedding).item()

print(f"Semantic similarity score: {cosine_similarity:.4f}")
