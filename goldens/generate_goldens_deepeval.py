import os, re, glob, json, random

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from deepeval.models import LocalModel, LiteLLMModel

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
#     task = 'text-generation',
#     max_new_tokens=500,
#     temperature=0.5
# )

# model = ChatHuggingFace(llm=llm)


model = LiteLLMModel(
    # model="groq/llama-3.1-8b-instant",
    # model="groq/openai/gpt-oss-20b",
    model = "groq/qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY")
)


# --- reuse your own VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in glob.glob("data/*.vtt"):
        with open(path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
sample = random.sample(chunks, min(15, len(chunks)))     # ~12 chunks -> keep the set small
contexts = [[c] for c in sample]                          # each context = one chunk

synthesizer = Synthesizer(model=model)                # the generator/critic model -- pin it
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,       # <-- THIS gives you the ideal_answer
    max_goldens_per_context=1,          # 1 question per chunk -> ~12 goldens
)

# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

with open("goldens/retriever_deepeval_goldens.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> goldens/component_goldens_draft.json")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")