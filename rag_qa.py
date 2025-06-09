"""
CLI usage:
    python rag_qa.py "Why did my firewall drop invalid packets?"
"""
import sys, pickle, faiss, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ─────────────── Configuration ───────────────
VECTOR_DIR = Path(r"vector_store")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_NAME    = "google/flan-t5-base"      # small, free, instruction-tuned
TOP_K = 8                                # # retrieved chunks
MAX_OUTPUT = 256                         # tokens
# ──────────────────────────────────────────────

# 1. Load vector index + docs
index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
docs  = pickle.load(open(VECTOR_DIR / "docs.pkl", "rb"))

# 2. Embedder
embedder = SentenceTransformer(EMBED_MODEL)

# 3. LLM (Seq2Seq, CPU-friendly)
tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME)
generate = pipeline("text2text-generation",
                    model=llm,
                    tokenizer=tok,
                    max_new_tokens=MAX_OUTPUT)

def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, k)
    return [docs[i] for i in idx[0]]

def build_prompt(context, question):
    return f"""
    You are a senior network reliability assistant. Answer the QUESTION using only the CONTEXT provided.

    Respond using **this format only**:

    Root Cause:
    <short summary of the issue, based on logs and explanations>

    Recommended Actions:
    1. <step 1>
    2. <step 2>
    3. <step 3>

    Evidence:
    <cite the most relevant error codes, timestamps, or message snippets from the CONTEXT>

    CONTINUE even if only partial information is found. Do not say "Not enough information."

    CONTEXT:
    {context}

    QUESTION: {question}
    """.strip()


def trim_context(context, max_chars=512):
    """Trim context to avoid exceeding token limits. max_chars ~ 512 tokens."""
    if len(context) <= max_chars:
        return context
    parts = context.split('---')
    trimmed = ''
    for p in parts:
        if len(trimmed) + len(p) + 5 > max_chars:
            break
        trimmed += p.strip() + "\n---\n"
    return trimmed.strip()


def answer(question):
    ctx = "\n---\n".join(retrieve(question))
    context = trim_context(ctx)
    prompt = build_prompt(context, question)
    out = generate(prompt)[0]["generated_text"]
    #print("\n🔍 Retrieved Chunks:")
    #for i, c in enumerate(ctx.split("---")):
    #    print(f"\nChunk {i+1}:\n{c.strip()[:300]}")
    return out.strip()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Explain NAT_TABLE_OVERFLOW"
    print("\n" + "="*60)
    print("QUESTION:", q)
    print("="*60)
    print(answer(q))
    print("="*60 + "\n")
