import pandas as pd
from transformers import pipeline

# Initialize sentiment-analysis and NLI pipelines
sentiment_model = pipeline("sentiment-analysis")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

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

        # Perform sentiment analysis
        review_sentiment = sentiment_model(review_text)[0]['label']
        summary_sentiment = sentiment_model(summary)[0]['label']
        
        # Perform NLI to check if the summary is faithful to the reviewText
        nli_result = nli_model(f"{review_text} [SEP] {summary}")[0]
        
        # Append results
        results.append({
            "reviewText": review_text,
            "rating": rating,
            "summary": summary,
            "review_sentiment": review_sentiment,
            "summary_sentiment": summary_sentiment,
            "hallucination": nli_result['label'],
            "hal_score": nli_result['score']
        })

    return results

# Load the Excel file
results = []
n_files = 25 # set it
ls = ['split_datasets/split_'+str(i+1)+'.csv' for i in range(n_files)]
for file in ls:
    results = results+main(file)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
# Output the results to a new Excel file
output_file_path = 'nli_sent.csv'
results_df.to_csv(output_file_path, index=False)
