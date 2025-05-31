import json
import os
from minio import Minio
from datetime import timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from typing import Dict, Any
import glob

load_dotenv()

# Azure OPENAI client
endpoint = os.getenv("ZRSVN_AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("ZRSVN_AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "moja.shramba.arnes.si"
bucket_name = "zrsvn-rag-najdbe"

s3_client = Minio(
    endpoint=s3_endpoint_url,
    access_key=s3_access_key,
    secret_key=s3_secret_access_key,
    secure=True  # True for HTTPS
)

# Load the source JSON data
data_folder = './preprocess_data'
all_files = glob.glob(os.path.join(data_folder, '*.json'))

def generate_presigned_url(file_key: str, page_number: int) -> str:
    """
    Generates a presigned URL for accessing a specific file and appends a page anchor.
    """
    try:
        presigned_url = s3_client.presigned_get_object(
            bucket_name,
            file_key,
            expires=timedelta(hours=1)  # Valid for 1 hour
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        return f"Error generating link: {e}"
    
# Pydantic model for the GPT response
# We generate two QA pairs for each text element
class QAPairs(BaseModel):
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
                Please format your answers concisely without extra newlines or excessive whitespace.
                Use only single spaces between words and avoid adding line breaks unless necessary.

                Important considerations for questions:
                - The question should be clear enough that a single answer can be found for it.
                - Ensure that every question is strictly based on the content of the provided text.
                - The questions should not introduce external context or assumptions. Stick only to the information available in the text.
                - Avoid questions that are too general or abstract. For example, instead of asking "What is this text about?", focus on specific facts or details mentioned in the text.
                - Avoid using phrases like "in this paragraph", "in this section", "in this document" in the questions.
                - Each question should target a specific concept or piece of information in the text.
                - The question must be a standalone, direct query and must never contain answers, explanations, or additional commentary.
                - Do not include multiple questions in a single question field; one question per entry only.
                - The question should be concise and clearly formulated to avoid ambiguity or confusion.
                - All of the generated text should be written in Slovenian language.

                Important considerations for answers:
                - The answer should directly address the question without introducing external context or assumptions.
                - Keep the answer concise, focusing on the most relevant and clear part of the content that answers the question.
                - Avoid unnecessary elaboration, explanations, or additional details that are not needed to answer the question.
                - If the answer is based on specific text from the provided input, use exact or paraphrased wording from that text. Do not invent or assume details outside of the provided information.
                - Each answer should be easy to understand and should not require additional interpretation. Avoid complex or vague responses.
                - Ensure that all answers are strictly based on the content of the text, providing only the most relevant information to answer the question at hand.
                - The answer must contain only the direct response to the question, never any additional questions or unrelated text.
                - Do not include any questions, prompts, or follow-up statements inside the answer field.
                - The answer must be a standalone, self-contained response that clearly and precisely addresses the question.
                - All of the generated text should be written in Slovenian language.'''
            },
            {
                "role": "user",
                "content": context
            },
        ],
        response_format=QAPairs
    )

    # 'parsed' is a QAPair object, so we convert it to a dict
    parsed_model = response.choices[0].message.parsed
    data = parsed_model.dict()

    transformed_data = {
        'questions_answers': [
            {'question': data[f'question_{i}'], 'answer': data[f'answer_{i}']}
            # This equals the number of QA pairs specified in QAPairs
            for i in range(1, 3)
        ]
    }
    return {
        **transformed_data,
        'text': gen_data['text'],
        'chunkID': gen_data['chunkID'],
        'fileUrl': gen_data['fileUrl'],
        'fileS3Path': gen_data['fileS3Path'],
        'fileName': gen_data['fileName'],
        'pageNumber': gen_data['pageNumber'],
        'boundingBox': gen_data.get('boundingBox', []),
    }

processed_data = []

# Configuration for maximum entries per page
# We choose to take a maximum of 2 text items per page
# which should each be at least 512 characters in size.
#
# For each of the text items we will later on generate 2 QA pairs.
# This will give a max total of 4 QA pairs per page.
for file_path in all_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_entries_per_page = 2
    num_current_entries_per_page = {}
    file_name = data['fileName']
    file_s3_path = data['fileS3Path']

    prepared_data = []

    # Populate prepared_data by iterating through the source JSON
    for page in data['documentPages']:
        page_number = page['pageNumber']

        for chunk in page['chunks']:
            # Skip short text.
            # Ideally we would be dealing with 2048 characters (taking into account having
            # 512 tokens as our chunk size and that 1 token roughly equals 4 characters).
            if chunk.get('nrCharacters') is None or chunk['nrCharacters'] < 512:
                continue

            if num_current_entries_per_page.get(page_number, 0) >= max_entries_per_page:
                continue

            presigned_url = generate_presigned_url(file_s3_path, page_number)
            chunk_id = chunk['chunkID']
            # Prepare the chunk for QA generation
            # Provide boundingBox if it exists, otherwise empty list
            bounding_box = chunk.get('boundingBox', [])

            prepared_data.append({
                **chunk,
                'fileUrl': presigned_url,
                'fileS3Path': file_s3_path,
                'fileName': file_name,
                'pageNumber': page_number,
            })

            num_current_entries_per_page[page_number] = num_current_entries_per_page.get(page_number, 0) + 1

    print(f"Processing QA pairs for {file_name}")
    
    # Generate QA pairs for each chunk in this file
    for entry in prepared_data:
        if not entry.get('text'):
            continue
        qa_output = generate_qa_pairs(entry)
        processed_data.append(qa_output)

# Save processed data as JSON
os.makedirs("app_data", exist_ok=True)
output_path = "app_data/qa_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"QA processing complete! Data saved to {output_path}.")