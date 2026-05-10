# ./kgs_builder/core/summerize.py

import os
import asyncio
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from kgs_builder.nano_graphrag._llm import gemini_complete_if_cache, _get_openrouter_model
from helpers.logger import get_logger
logger = get_logger("retrieve", log_file="logs/retrieve.log")


sum_prompt = """
Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

ANATOMICAL_STRUCTURE: Mention any anatomical structures specifically discussed.
BODY_FUNCTION: List any body functions highlighted.
BODY_MEASUREMENT: Include normal measurements like blood pressure or temperature.
BM_RESULT: Results of these measurements.
BM_UNIT: Units for each measurement.
BM_VALUE: Values of these measurements.
LABORATORY_DATA: Outline any laboratory tests mentioned.
LAB_RESULT: Outcomes of these tests (e.g., 'increased', 'decreased').
LAB_VALUE: Specific values from the tests.
LAB_UNIT: Units of measurement for these values.
MEDICINE: Name medications discussed.
MED_DOSE, MED_DURATION, MED_FORM, MED_FREQUENCY, MED_ROUTE, MED_STATUS, MED_STRENGTH, MED_UNIT, MED_TOTALDOSE: Provide concise details for each medication attribute.
PROBLEM: Identify any medical conditions or findings.
PROCEDURE: Describe any procedures.
PROCEDURE_RESULT: Outcomes of these procedures.
PROC_METHOD: Methods used.
SEVERITY: Severity of the conditions mentioned.
MEDICAL_DEVICE: List any medical devices used.
SUBSTANCE_ABUSE: Note any substance abuse mentioned.
Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.
"""


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


async def call_api(chunk):
    provider = os.getenv("LLM_PROVIDER") or "openrouter"
    model = _get_openrouter_model()
    return await gemini_complete_if_cache(
        model=model,
        prompt=chunk,
        system_prompt=sum_prompt,
        provider=provider,
        temperature=0.2,
        max_tokens=1000,
        n=1,
        stop=None,
    )

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunk_words = words[i:i+tokens]
        chunk_text = encoding.decode(chunk_words)
        chunks.append(chunk_text)
    return chunks

async def process_chunks_async(content, max_concurrency=5):
    chunks = split_into_chunks(content)
    if not chunks:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _safe_call(chunk, index):
        async with semaphore:
            try:
                return await call_api(chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {index+1}/{len(chunks)}: {e}")
                return ""

    tasks = [_safe_call(chunk, i) for i, chunk in enumerate(chunks)]
    return await asyncio.gather(*tasks)


def process_chunks(content):
    return _run_async(process_chunks_async(content))

if __name__ == "__main__":
    content = "meomeo cute nhat the gioi"
    responses = process_chunks(content)
    for resp in responses:
        print(resp)
