import json
import os
import hashlib
import requests
import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "https://moja.shramba.arnes.si"
bucket_name = "zrsvn-rag-najdbe"

s3_client = boto3.client(
    's3',
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_access_key
)

# Load the source JSON data
data_path = './preprocess_data/new_22 ZRC SAZU_PoLJUBA_Reintrodukcija_2Poročilo_23c194f1_4df89569.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Configuration for maximum entries per page
max_per_page = 2
num_per_page = {}
file_name = data['fileName']
file_s3_path = data['fileS3Path']

qa_generation_data = []

def generate_presigned_url(file_key: str, page_number: int) -> str:
    """
    Generates a presigned URL for accessing a specific file and appends a page anchor.
    """
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_key},
            ExpiresIn=3600  # Valid for 1 hour
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        return f"Error generating link: {e}"

# Populate qa_generation_data by iterating through the source JSON
for page in data['documentPages']:
    page_number = page['pageNumber']

    for chunk in page['chunks']:
        # Skip short text
        if chunk.get('nrCharacters') is None or chunk['nrCharacters'] < 512:
            continue

        if num_per_page.get(page_number, 0) >= max_per_page:
            continue

        presigned_url = generate_presigned_url(file_s3_path, page_number)

        chunk_id = chunk['chunkID']

        # Prepare the chunk for QA generation
        # Provide boundingBoxes if they exist, otherwise empty list
        bounding_boxes = chunk.get('boundingBoxes', [])

        qa_generation_data.append({
            **chunk,
            'documentUrl': presigned_url,
            'pageNumber': page_number,
            'fileName': file_name,
        })

        num_per_page[page_number] = num_per_page.get(page_number, 0) + 1

# Pydantic model for the GPT response
class QAModel(BaseModel):
    question_1: str
    answer_1: str
    question_2: str
    answer_2: str

# QA pair generation
def generate_qa_pairs(gen_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates QA pairs using a specialized GPT-4o-mini method.
    """
    # Build the prompt context
    context = f"Use the following text to generate question answer pairs:\n\n{gen_data['text']}"

    # Make the request to LLM
    response = client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": '''You are an AI assistant that always responds in Slovene.
                You are tasked with turning text into a set of question-answer pairs. 
                The goal is to create a set of clear, specific, and relevant questions and answers 
                that can be answered directly from the provided text.

                Important considerations for questions:
                - The question should be clear enough that a single answer can be found for it.
                - Ensure that every question is strictly based on the content of the provided text.
                - The questions should not introduce external context or assumptions. Stick only to the information available in the text.
                - Avoid questions that are too general or abstract. For example, instead of asking "What is this text about?", focus on specific facts or details mentioned in the text.
                - Avoid using phrases like "in this paragraph", "in this section", "in this document" in the questions.
                - Each question should target a specific concept or piece of information in the text.
                - All of the generated text should be written in Slovenian language.

                Important considerations for answers:
                - The answer should directly address the question without introducing external context or assumptions.
                - Keep the answer concise, focusing on the most relevant and clear part of the content that answers the question.
                - Avoid unnecessary elaboration, explanations, or additional details that are not needed to answer the question.
                - If the answer is based on specific text from the provided input, use exact or paraphrased wording from that text. Do not invent or assume details outside of the provided information.
                - Each answer should be easy to understand and should not require additional interpretation. Avoid complex or vague responses.
                - Ensure that all answers are strictly based on the content of the text, providing only the most relevant information to answer the question at hand.
                - All of the generated text should be written in Slovenian language.'''
            },
            {
                "role": "user",
                "content": context
            },
        ],
        response_format=QAModel
    )

    # 'parsed' is a QAModel object, so convert it to a dict:
    parsed_model = response.choices[0].message.parsed
    data = parsed_model.dict()

    transformed_data = {
        #'rationale': data['rationale'],
        'questions_answers': [
            {'question': data[f'question_{i}'], 'answer': data[f'answer_{i}']}
            for i in range(1, 3)
        ]
    }
    return {
        **transformed_data,
        'documentUrl': gen_data['documentUrl'],
        'pageNumber': gen_data['pageNumber'],
        'boundingBoxes': gen_data.get('boundingBoxes', []),
        'chunkID': gen_data['chunkID'],
        'text': gen_data['text'],
    }

# Generate QA pairs for each chunk
processed_data = []
for entry in qa_generation_data:
    # Some chunks might not have 'suggestedText' – handle gracefully
    if not entry.get('text'):
        continue
    qa_output = generate_qa_pairs(entry)
    processed_data.append(qa_output)

# Save processed data as JSON
os.makedirs("data", exist_ok=True)
output_path = "data/processed_qa_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"QA processing complete! Data saved to {output_path}.")
