import os
import time
import pandas as pd
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key='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'  # 确保使用安全方式管理API密钥
)

data_path = './data/split_datasets/split_1.csv'

# 从CSV文件中读取数据
data = pd.read_csv(data_path)


# 定义函数以生成对整个文件的综合评价
def generate_comprehensive_review(data):
    reviews_text = "\n".join(data['reviewText'].astype(str))  # 将所有评论文本合并为一个大的字符串
    summaries_text = "\n".join(data['summary'].astype(str))  # 将所有摘要合并为一个大的字符串
    overall_ratings = data['overall'].mean()  # 计算平均评分

    # 构建输入文本
    input_text = f"Collected Review Texts:\n{reviews_text}\n\nCollected Summaries:\n{summaries_text}\n\nAverage Rating: {overall_ratings}\n###\n\nBased on the collected review texts, summaries, and the average rating, please provide a comprehensive analysis of the overall customer satisfaction and product quality."

    # 请求模型生成文本
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": """You are an assistant who specializes in analyzing user reviews and providing structured input. Please provide a detailed analysis based on the provided review text, overall ratings and summaries and answer in the following format:
1. Review Text: Analyze the detailed opinions and emotional tendencies of users based on the review text.
2. Overall Rating: Explain what aspects of the product users are satisfied or dissatisfied with as reflected in the ratings provided.
3. Summary: Combine the above information to provide a summary opinion of the product, identifying key strengths and potential areas for improvement.
Make sure your answer is clear, organized, and responds to each section in detail.
"""},
                {"role": "user", "content": input_text}
            ],
            max_tokens=500
        )
        return response.choices[0].message
    except Exception as e:
        if 'insufficient_quota' in str(e):
            print("Rate limit exceeded, waiting 10 s...")
            time.sleep(10)
            return generate_comprehensive_review(data)  # Retry
        else:
            print(f"An error occurred: {e}")
            return None


# 生成综合评论并保存到文件
comprehensive_review = generate_comprehensive_review(data)
if comprehensive_review:
    with open('output_reviews_v3.txt', 'w') as outfile:
        outfile.write(f"{comprehensive_review}\n\n")
