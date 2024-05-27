import pandas as pd
from transformers import pipeline

# Load the Excel file
file_path = 'split_datasets/split_1.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Initialize sentiment-analysis and NLI pipelines
sentiment_model = pipeline("sentiment-analysis")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Analyze the data and collect results
results = []
for index, row in data.iterrows():
    review_text = row['reviewText']
    rating = row['overall']
    summary = row['summary']

    # Perform sentiment analysis
    review_sentiment = sentiment_model(review_text)[0]['label']
    summary_sentiment = sentiment_model(summary)[0]['label']
    consistency = (review_sentiment == summary_sentiment)
    
    # Perform NLI to check if the summary is faithful to the reviewText
    nli_result = nli_model(f"{review_text} [SEP] {summary}")[0]
    
    # Append results
    results.append({
        "reviewText": review_text,
        "rating": rating,
        "summary": summary,
        "review_sentiment": review_sentiment,
        "summary_sentiment": summary_sentiment,
        "consistency": consistency,
        "hallucination": nli_result['label'],
        "hal_score": nli_result['score']
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Output the results to a new Excel file
output_file_path = 'split_1_nli.csv'
results_df.to_csv(output_file_path, index=False)
