# ./kgs_builder/core/agentic_chunker.py

import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from nano_graphrag._llm import _get_openrouter_client

class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5

        # whether or not to update/refine summaries and titles as new propositions arrive
        self.generate_new_metadata_ind = True
        self.print_logging = True

        self.model = "google/gemini-2.0-flash-lite-001"
        self.max_retries = 3
        self.retry_sleep_seconds = 1.0

        self.llm_client = _get_openrouter_client()

    def _run_async(self, coro):
        """Run an async coroutine from sync methods safely."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        # If already in an event loop (e.g. notebook), run in a worker thread.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    async def _call_gemini_once(self, prompt: str) -> str:
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _call_gemini_with_retry(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._run_async(self._call_gemini_once(prompt)).strip()
            except Exception as e:
                last_error = e
                if self.print_logging:
                    print(f"LLM call failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        system_prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a group of propositions which are in the chunk and the chunks current summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the chunk new summary, nothing else.
        """
        
        user_prompt = f"Chunk's propositions:\n{chr(10).join(chunk['propositions'])}\n\nCurrent chunk summary:\n{chunk['summary']}"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        new_chunk_summary = self._call_gemini_with_retry(full_prompt)
        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        system_prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

        A good title will say what the chunk is about.

        You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

        Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.
        """
        
        user_prompt = f"Chunk's propositions:\n{chr(10).join(chunk['propositions'])}\n\nChunk summary:\n{chunk['summary']}\n\nCurrent chunk title:\n{chunk['title']}"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        updated_chunk_title = self._call_gemini_with_retry(full_prompt)
        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        system_prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the new chunk summary, nothing else.
        """
        
        user_prompt = f"Determine the summary of the new chunk that this proposition will go into:\n{proposition}"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        new_chunk_summary = self._call_gemini_with_retry(full_prompt)
        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        system_prompt = """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

        A good chunk title is brief but encompasses what the chunk is about

        You will be given a summary of a chunk which needs a title

        Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.
        """
        
        user_prompt = f"Determine the title of the chunk that this summary belongs to:\n{summary}"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        new_chunk_title = self._call_gemini_with_retry(full_prompt)
        return new_chunk_title


    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        system_prompt = """
        Determine whether or not the "Proposition" should belong to any of the existing chunks.

        A proposition should belong to a chunk of their meaning, direction, or intention are similar.
        The goal is to group similar propositions and chunks.

        If you think a proposition should be joined with a chunk, return the chunk id.
        If you do not think an item should be joined with an existing chunk, just return "No chunks"

        Example:
        Input:
            - Proposition: "Greg really likes hamburgers"
            - Current Chunks:
                - Chunk ID: 2n4l3d
                - Chunk Name: Places in San Francisco
                - Chunk Summary: Overview of the things to do with San Francisco Places

                - Chunk ID: 93833k
                - Chunk Name: Food Greg likes
                - Chunk Summary: Lists of the food and dishes that Greg likes
        Output: 93833k
        """
        
        user_prompt = f"Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--\n\nDetermine if the following statement should belong to one of the chunks outlined:\n{proposition}"
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        chunk_found = self._call_gemini_with_retry(full_prompt)

        # Try to extract chunk id from response
        # Look for a string that matches the id length
        import re
        id_pattern = r'\b[a-zA-Z0-9]{' + str(self.id_truncate_limit) + r'}\b'
        match = re.search(id_pattern, str(chunk_found))
        
        if match:
            chunk_found = match.group(0)
            # Verify this chunk id exists
            if chunk_found in self.chunks:
                return chunk_found
        
        return None
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print (f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self.get_chunk_outline())

if __name__ == "__main__":
    ac = AgenticChunker()

    ## Comment and uncomment the propositions to your hearts content
    propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
        # 'Teachers and coaches meant well.',
        # "The statement that 'You get out what you put in' is rarely true.",
        # "If your product is only half as good as your competitor's product, you do not get half as many customers.",
        # "You get no customers if your product is only half as good as your competitor's product.",
        # 'You go out of business if you get no customers.',
        # 'The returns for performance are superlinear in business.',
        # 'Some people think the superlinear returns for performance are a flaw of capitalism.',
        # 'Some people think that changing the rules of capitalism would stop the superlinear returns for performance from being true.',
        # 'Superlinear returns for performance are a feature of the world.',
        # 'Superlinear returns for performance are not an artifact of rules that humans have invented.',
        # 'The same pattern of superlinear returns is observed in fame.',
        # 'The same pattern of superlinear returns is observed in power.',
        # 'The same pattern of superlinear returns is observed in military victories.',
        # 'The same pattern of superlinear returns is observed in knowledge.',
        # 'The same pattern of superlinear returns is observed in benefit to humanity.',
        # 'In fame, power, military victories, knowledge, and benefit to humanity, the rich get richer.'
    ]
    
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print (ac.get_chunks(get_type='list_of_strings'))