# Ground Truth QA Evaluation

Sistem za generiranje in vrednotenje parov vprašanj in odgovorov iz PDF dokumentov z uporabo LLMov.

## O projektu

Sistem implementira dvo-fazni pristop za ustvarjanje kakovostnih QA parov:  
- **Faza 1:** Avtomatsko generiranje parov vprašanj in odgovorov iz besedilnih odsekov PDF dokumentov z LLMom (privzeta izbira je Azure OpenAI - GPT-4o-mini).  
- **Faza 2:** Človeška evaluacija in popravki generiranih parov prek spletnega vmesnika.

> **Osnovo za razvoj te aplikacije je predstavljal Jupyter notebook podjetja [EyeLevel](https://www.eyelevel.ai/), ki je dostopen tu:**  
> https://github.com/groundxai/code-samples/blob/master/notebooks/RAGMasters_QAGenWithHuman.ipynb

## Funkcionalnosti

- Obdelava JSON datotek z besedilnimi odseki iz PDF dokumentov.
- Generiranje 2 parov vprašanj in odgovorov na besedilni odsek z LLMom.
- Varno povezovanje do PDF dokumentov prek S3/MinIO shrambe.
- Spletni vmesnik za sekvenčno vrednotenje QA parov.
- Prikaz PDF strani z označenimi robnimi okvirji besedilnih odsekov.
- Možnost urejanja in popravljanja generiranih vprašanj in odgovorov.
- Shranjevanje povratnih informacij za kasnejšo analizo in možnost evaluacije samih RAG sistemov.

## Tehnične zahteve

- Python 3.8+.
- FastAPI za spletni strežnik.
- Azure OpenAI API dostop.
- MinIO/S3 shramba za PDF dokumente.
- PyMuPDF (fitz) za renderiranje PDF strani.
- PIL/Pillow za obdelavo slik.

## Namestitev

1. Klonirajte repozitorij:
    ```bash
    git clone https://github.com/gregorgatej/ground-truth-qa.git
    cd ground-truth-qa
    ```

2. Namestite odvisnosti:
    ```bash
    pip install fastapi uvicorn python-dotenv minio openai pydantic PyMuPDF pillow requests jinja2
    ```

3. Ustvarite `.env` datoteko z naslednjimi spremenljivkami:
    ```
    ZRSVN_AZURE_OPENAI_ENDPOINT=tvoj_azure_endpoint
    ZRSVN_AZURE_OPENAI_KEY=tvoj_azure_openai_key
    S3_ACCESS_KEY=tvoj_s3_access_key
    S3_SECRET_ACCESS_KEY=tvoj_s3_secret_key
    ```

4. Pripravite strukturo map:
    ```bash
    mkdir -p preprocess_data app_data static assets templates
    ```

## Uporaba

### 1. Predprocesiranje podatkov

Postavite JSON datoteke (tj. rezultat prve faze procesa predprocesiranja v [zrsvn-rag-preprocessing](https://github.com/gregorgatej/zrsvn-rag-preprocessing)) z besedilnimi odseki v mapo `preprocess_data/` in zaženite:
```bash
python preprocess.py
```

Skripta bo:

- Obdelala vse JSON datoteke v mapi preprocess_data/
- Za vsak besedilni odsek (min. 512 znakov) generirala 2 para vprašanj in odgovorov.
- Rezultate shranila v app_data/qa_data.json.

### 2. Spletna evaluacija

Zaženite spletni strežnik:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Pregledovalec QA parov lahko sedaj odpre brskalnik na http://localhost:8000 in:

- Pregleda prikazane pare vprašanj in odgovorov.
- Jih označi kot "Ustrezen", "Neustrezen" ali "Preskoči".
- Po potrebi uredi in popravi njihovo vsebino.
- Povratne informacije se shranjujejo v app_data/feedback.json.