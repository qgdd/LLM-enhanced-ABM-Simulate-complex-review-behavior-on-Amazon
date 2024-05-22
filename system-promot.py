import os
import time
import pandas as pd
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key='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'  # this is also the default, it can be omitted
)


# 定义生成评价的函数
def generate_review(review_text, overall, summary):
    input_text = f"Review Text: {review_text}\nOverall Rating: {overall}\nSummary: {summary}\n###\n\nPlease provide a comprehensive analysis based on the review text, overall rating, and summary provided."
    # 请求模型生成文本
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that provides opinions on products."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=150,
        )
    except Exception as e:
        if 'insufficient_quota' in str(e):
            print("Rate limit exceeded, waiting 10 s...")
            time.sleep(10)  # Wait for 10 s
            return generate_review(review_text, overall, summary)  # Retry
        else:
            print(f"An error occurred: {e}")
            return None
    return response.choices[0].message # 保证提取正确的回复内容


def process_files(data_folder):
    # 创建结果文件
    with open('output_reviews- v1.txt', 'w') as outfile:
        # 遍历每个分割的文件
        for filename in sorted(os.listdir(data_folder)):
            if filename.startswith('split') and filename.endswith('.csv'):
                print(f"Processing {filename}...")
                data = pd.read_csv(os.path.join(data_folder, filename), delimiter=',')

                # 遍历每条记录
                for index, row in data.iterrows():
                    review = generate_review(row['reviewText'], row['overall'], row['summary'])
                    if review:
                        outfile.write(f"{review}\n\n")


# 处理文件夹中的所有分割文件
process_files('./data/split_datasets')