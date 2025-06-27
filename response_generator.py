from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
import torch
import os


class ResponseGenerator:
    def __init__(self):
        llama_32 = "meta-llama/Llama-3.2-3B-Instruct"
        self.generator = pipeline(model=llama_32, device="cuda", torch_dtype=torch.bfloat16)

    def prepare_prompt(self, query_text, matching_chunks):
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in matching_chunks])
        prompt = [
          {"role": "system", "content": "Answer the question based only on the following context. If the answer cannot be found, reply: 'The context does not contain enough information.\n\n'" + context + "\n\n"},
          {"role": "user", "content": query_text},
        ]
        return prompt

    def predict(self, prompt):
        generation = self.generator(
          prompt,
          do_sample=False,
          temperature=1.0,
          top_p=1,
          max_new_tokens=50
        )
        response = self.generator(prompt, max_new_tokens=512, do_sample=False)
        return response[0]['generated_text']

    def format_response(self, response, matching_chunks):
        sources = set([doc.metadata.get("source", None) for doc, _score in matching_chunks])
        sources_file_names = [os.path.basename(source) for source in sources]
        delimiter=", "
        formatted_response = f"Response: {response[-1]['content']}\nSources: {delimiter.join(sources)}"
        return formatted_response