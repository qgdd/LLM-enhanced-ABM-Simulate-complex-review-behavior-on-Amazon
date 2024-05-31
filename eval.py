import pandas as pd
from transformers import pipeline
from rouge_metric import PerlRouge
from sentence_transformers import SentenceTransformer, util

# Initialize sentiment-analysis and NLI pipelines
sentiment_model = pipeline("sentiment-analysis")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Initialize the ROUGE metric
rouge = PerlRouge(rouge_n_max=2, rouge_l=True, rouge_w=True,
    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def main(file_path):
    data = pd.read_csv(file_path)

    # Analyze the data and collect results
    results = []
    for index, row in data.iterrows():
        
        review_text = row['reviewText']
        rating = row['overall']
        summary = row['summary']

        if pd.isna(review_text) or pd.isna(rating) or pd.isna(summary):
            continue
        else:
            review = row['reviewText']+' Rating is '+str(row['overall'])+' out of 5.'

        # Perform sentiment analysis
        review_sentiment = sentiment_model(review)[0]['label']
        summary_sentiment = sentiment_model(summary)[0]['label']
        
        # Perform NLI to check if the summary is faithful to the reviewText
        nli_result = nli_model(f"{review} [SEP] {summary}")[0]
        hallu = nli_result['label']
        halsc = nli_result['score']

        # Evaluate the summary using the ROUGE metric
        scores = rouge.evaluate([summary], [[review]])
        r1 = scores['rouge-1']['r']
        r2 = scores['rouge-2']['r']
        l = scores['rouge-l']['r']

        # Semantic embedding cosine similarity
        review_emb = model.encode(review, convert_to_tensor=True)
        summary_emb = model.encode(summary, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(review_emb, summary_emb).item()
        
        # Append results
        results.append({
            "reviewText": review_text,
            "rating": rating,
            "summary": summary,
            "review_sentiment": review_sentiment,
            "summary_sentiment": summary_sentiment,
            "hallucination": hallu,
            "hal_score": halsc,
            "rouge-1": r1,
            "rouge-2": r2,
            "rouge-l": l,
            "semantic_sim": cosine_similarity
        })

    return results

# Load the Excel file
results = []
n_files = 25
ls = ['split_datasets/split_'+str(i+1)+'.csv' for i in range(n_files)]
for file in ls:
    results = results+main(file)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Output the results to a new Excel file
output_file_path = 'eval.csv'
results_df.to_csv(output_file_path, index=False)
