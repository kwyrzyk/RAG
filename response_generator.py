from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
import os

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class ResponseGenerator:
    @staticmethod
    def prepare_prompt(query_text, matching_chunks):
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in matching_chunks])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(question=query_text, context=context)
        return prompt

    @staticmethod
    def predict(prompt):
        generator = pipeline("text-generation", model="google/gemma-2b-it", device=0)
        response = generator(prompt, max_new_tokens=128, do_sample=False)
        return response[0]['generated_text']

    @staticmethod
    def format_response(response, matching_chunks):
        sources = set([doc.metadata.get("source", None) for doc, _score in matching_chunks])
        sources_file_names = [os.path.basename(source) for source in sources]
        delimiter = ", "
        formatted_response = f"Response: {response}\nSources: {delimiter.join(sources_file_names)}"
        return formatted_response