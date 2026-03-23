# ./kgs_builder/data_processing/data_chunk.py

import json
from typing import List
from pydantic import BaseModel
from kgs_builder.core.agentic_chunker import AgenticChunker
from nano_graphrag._llm import _get_openrouter_client
from helpers.logger import get_logger

logger = get_logger("data_chunk", log_file="logs/data_chunk.log")


class Sentence(BaseModel):
    sentences: List[str]


def _strip_json_fences(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines).strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return content


def _extract_propositions(response_text: str) -> List[str]:
    cleaned = _strip_json_fences(response_text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            parsed = Sentence(**payload)
            return [s.strip() for s in parsed.sentences if s and s.strip()]
    except Exception:
        pass

    return [s.strip() for s in response_text.split("\n") if s.strip()]


async def run_chunk(essay):
    paragraphs = [p.strip() for p in essay.split("\n\n") if p and p.strip()]
    essay_propositions = []
    ac = AgenticChunker()
    client = _get_openrouter_client()

    for i, para in enumerate(paragraphs):
        try:
            prompt = f"""
            Extract main propositions from the following text. 
            Return strictly in JSON format with a "sentences" field containing an array of strings.

            Text:
            {para}
            
            Example format:
            {{"sentences": ["proposition 1", "proposition 2", ...]}}
            """

            response = await client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = (response.choices[0].message.content or "").strip()

            propositions = _extract_propositions(response_text)
            essay_propositions.extend(propositions)

            logger.info(f"Paragraph {i+1}/{len(paragraphs)}: Extracted {len(essay_propositions)} propositions so far.")

        except Exception as e:
            logger.error(f"Error processing paragraph {i+1}: {str(e)}")

    if essay_propositions:
        ac.add_propositions(essay_propositions)

    chunks = ac.get_chunks(get_type="list_of_strings")
    return chunks