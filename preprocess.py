# preprocess.py

import json
import os
import hashlib
import requests
import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any

# 1. Load environment variables
load_dotenv()

# 2. Create OpenAI client (replace with your actual OpenAI API key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3. Initialize S3 client
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "https://moja.shramba.arnes.si"
bucket_name = "eval"

s3_client = boto3.client(
    's3',
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_access_key
)

# 4. Load the source JSON data
data_path = './json_data/processed_SUGEXTRASPLITPoročilo VIPAVA Ph teleius 2021-11-20_split_sectionNumbers-2-10.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 5. Configuration for maximum entries per page
max_per_page = 2
num_per_page = {}
file_name = data['fileName']

# We'll store all QA generation input data here
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

# 6. Populate qa_generation_data by iterating through your JSON
for page in data['documentPages']:
    page_number = page['pageNumber']

    for chunk in page['chunks']:
        # Skip short text
        if chunk.get('nrCharacters') is None or chunk['nrCharacters'] < 500:
            continue

        if num_per_page.get(page_number, 0) >= max_per_page:
            continue

        presigned_url = generate_presigned_url(file_name, page_number)

        # Prepare the chunk for QA generation
        # Provide boundingBoxes if they exist, otherwise empty list
        bounding_boxes = chunk.get('boundingBoxes', [])

        qa_generation_data.append({
            **chunk,
            'documentUrl': presigned_url,
            'pageNumber': page_number,
            'fileName': file_name,
            'boundingBoxes': bounding_boxes
        })

        num_per_page[page_number] = num_per_page.get(page_number, 0) + 1

# 7. Pydantic model for the GPT response
class QAModel(BaseModel):
    rationale: str
    question_1: str
    answer_1: str
    question_2: str
    answer_2: str
    question_3: str
    answer_3: str
    question_4: str
    answer_4: str
    question_5: str
    answer_5: str


# 8. GPT QA pair generation
def generate_qa_pairs(gen_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates QA pairs using a specialized GPT-4o-mini method.
    (Adapted from your code that uses .parse)
    """
    # Build the prompt context
    context = f"Use the following data to generate question answer pairs:\n\n{gen_data['suggestedText']}"

    # Make the request to your specialized GPT model
    response = client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": '''You are an AI assistant that always responds in Slovene.
                You are tasked with turning data into
                a set of question answer pairs which will be used to test information
                extraction systems. it is your job to create a set of question and
                answer pairs which explore a range of topics concerned with the population count of a
                Carpodacus erythrinus bird, where the information is presented in several ways.

                First, look at the data and construct a "rationale". This rationale
                should include a breakdown of the types of information presented, and
                what an information extraction system might fail at when analyzing this
                type of data. Then, generate a batch of question-answer pairs
                that would adequetly test based on the rationale.

                Important considerations:
                - The question should be clear enough where a single answer can be found
                - There are tens of documents. Questions must be specific.
                - Avoid questions like "this figure", "in the section" and "in the document",
                as there are many documents.
                - The objective is to ask fair questions where a clear answer is obvious.
                - All the generated text should be written in Slovenian language'''
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
        'rationale': data['rationale'],
        'questions_answers': [
            {'question': data[f'question_{i}'], 'answer': data[f'answer_{i}']}
            for i in range(1, 6)
        ]
    }
    return {
        **transformed_data,
        'documentUrl': gen_data['documentUrl'],
        'pageNumber': gen_data['pageNumber'],
        'boundingBoxes': gen_data.get('boundingBoxes', [])
    }

# 9. Generate QA pairs for each chunk
processed_data = []
for entry in qa_generation_data:
    # Some chunks might not have 'suggestedText' – handle gracefully
    if not entry.get('suggestedText'):
        continue
    qa_output = generate_qa_pairs(entry)
    processed_data.append(qa_output)

# 10. Save processed data as JSON for the Flask app
os.makedirs("data", exist_ok=True)
output_path = "data/processed_qa_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"QA processing complete! Data saved to {output_path}.")
