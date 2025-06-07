# Obdelava JSON podatkov iz ene ali več datotek
# (ki so rezultat skript v zrsvn-rag-preprocessing oz.
# katerakoli datoteka iz mape output_jsons). 
# Za vsak 'text' pod 'chunks' znotraj JSON
# datoteke, ki je dovoljšnje velikosti samodejno generira določeno število parov
# vprašanj in odgovorov, s pomočjo izbranega LLMa.
# Rezultat skripte so pari vprašanj in odgovorov, 
# ki se shranijo v skupno izhodno datoteko
# (app_data/qa_data.json). Slednjo uporablja app.py.

# Delo z datotekami in mapami.
import os
import glob
import json
# Branje podatkov iz .env datoteke.
from dotenv import load_dotenv
# Povezava z oddaljenim strežnikom (tj. z S3 kompatibilna shramba) in
# generiranje varnih povezav do PDF datotek. 
from minio import Minio
# Delo z roki veljavnosti povezav do datotek.
from datetime import timedelta
# Klici LLMu, ki generira pare vprašanj in odgovorov.
from openai import AzureOpenAI
# Preverjanje in zagotavljanje pravilne oblike izhodnih podatkov.
from pydantic import BaseModel
# Boljša berljivost tipov vhodnih podatkov.
from typing import Dict, Any

# Preberemo varnostne ključe in parametre.
load_dotenv()

# Vzpostavimo povezavo do storitve Azure OpenAI.
endpoint = os.getenv("ZRSVN_AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("ZRSVN_AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Vzpostavimo povezavo do S3 shrambe prek Minio klienta.
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "moja.shramba.arnes.si"
bucket_name = "zrsvn-rag-najdbe"

s3_client = Minio(
    endpoint=s3_endpoint_url,
    access_key=s3_access_key,
    secret_key=s3_secret_access_key,
    secure=True
)

# Nastavimo poti do vseh JSON datotek znotraj izbrane mape.
data_folder = './preprocess_data'
all_files = glob.glob(os.path.join(data_folder, '*.json'))

# Naredi varno povezavo do izbrane datoteke na S3 strežniku, ki velja
# 1 uro ter doda oznako za določeno stran v PDFju (npr. #page=5).
def generate_presigned_url(file_key: str, page_number: int) -> str:
    try:
        presigned_url = s3_client.presigned_get_object(
            bucket_name,
            file_key,
            # Veljavnost povezave bo 1 uro.
            expires=timedelta(hours=1)
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        return f"Error generating link: {e}"
    
# S pomočjo Pydantica definiramo razred, ki bo poskrbel, da bo format
# odgovora od LLMa vedno vseboval 2 vprašanji in 2 odgovora.
class QAPairs(BaseModel):
    question_1: str
    answer_1: str
    question_2: str
    answer_2: str

# Generiramo pare vprašanj in odgovorov, ki temeljijo na danem besedilu.
# Izhod klica LLMu je nastavljen tako, da se pričakuje izhod v obliki kot jo
# določa Pydantic model QAPairs.
# Rezultat se preoblikuje v standardizirano podatkovno obliko slovarja 
# (ang. dictionary) in poleg shrani pomembne dodatne informacije (identifikator,
# številka strani, povezava do dokumenta itd.).
def generate_qa_pairs(gen_data: Dict[str, Any]) -> Dict[str, Any]:
    # Na podlagi vhodnega teksta pripravimo del poziva, ki bo nudil LLMu kontekst
    # na podlagi katerega bo pripravil odgovore. 
    context = f"Use the following text to generate question answer pairs:\n\n{gen_data['text']}"

    # Pošljemo zahtevo LLMu.
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

    # 'parsed' je objekt tipa QAPairs, zato ga pretvorimo v slovar.
    parsed_model = response.choices[0].message.parsed
    data = parsed_model.dict()

    transformed_data = {
        'questions_answers': [
            {'question': data[f'question_{i}'], 'answer': data[f'answer_{i}']}
            # Spodnja vrednost je enaka številu parov vprašanje-odgovor,
            # navedenih v razredu QAPairs.
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

# Seznam v katerega bomo shranjevali rezultate klicov funkciji
# generate_qa_pairs.
processed_data = []

# Glavna zanka za obdelavo JSON datotek, prek katere pretvorimo podatke
# v obliko, ki je primerna kot vhod funkciji generate_qa_pairs.
# Za vsako stran navedeno v datoteki obravnavamo največ 2 besedilna odseka (ang. chunk),
# ki morata biti dovolj dolga, tj. imeti vsaj 512 znakov.
# Za vsak besedilni odsek:
# - Ustvarimo varno povezavo, ki vodi do strani v izvornem
#   PDF dokumentu kjer se pojavi.
# - Dodamo pomembne metapodatke.
# Ker bosta za vsakega izmed besedilnih odsekov generirana 2 para vprašanj
# in odgovorov bomo na koncu zagona naše skripte pridobili rezultat,
# ki bo vseboval največ 4 pare vprašanj in odgovorov na posamezno stran
# izvornega PDF dokumenta.
for file_path in all_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_entries_per_page = 2
    num_current_entries_per_page = {}
    file_name = data['fileName']
    file_s3_path = data['fileS3Path']

    # Seznam v katerega shranjujemo podatke, ki bodo predstavljali
    # vhod funkciji generate_qa_pairs.
    prepared_data = []

    # Iteriramo skozi vhodno JSON datoteko, da zapolnimo prepared_data s
    # podatki, ki jih potrebuje funkcija generate_qa_pairs.
    for page in data['documentPages']:
        page_number = page['pageNumber']

        for chunk in page['chunks']:
            # Krajši tekst preskočimo.
            # V idealnih okoliščinah bi si želeli imeti opravka s čim več
            # besedilnimi odseki dolžine vsaj 2048 znakov (če vzamemo v zakup,
            # da imamo v glavni zrsvn-rag aplikaciji nastavljenih 512 tokenov kot velikost
            # besedilnega bloka (ang. chunk size) in da se en token enači s približno štirimi znaki). 
            if chunk.get('nrCharacters') is None or chunk['nrCharacters'] < 512:
                continue

            if num_current_entries_per_page.get(page_number, 0) >= max_entries_per_page:
                continue

            presigned_url = generate_presigned_url(file_s3_path, page_number)
            chunk_id = chunk['chunkID']
            # Dodamo mere robnega okvirja (ang. bounding box), če ta za
            # izbrani zapis obstaja,
            # sicer dodamo prazen seznam.
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
    
    # Generiramo pare vprašanj in odgovorov za vsakega izmed tekstualnih elementov
    # znotraj prepared_data seznama.
    # Rezultati se shranjujejo v skupni seznam imenovan processed_data.
    for entry in prepared_data:
        if not entry.get('text'):
            continue
        qa_output = generate_qa_pairs(entry)
        processed_data.append(qa_output)

# Rezultat shranimo v novo datoteko oblike JSON.
os.makedirs("app_data", exist_ok=True)
output_path = "app_data/qa_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"QA processing complete! Data saved to {output_path}.")