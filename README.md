# Charter communication RAG Assistant – RAG-based Log Analyzer with Open-Source LLM

This project is a lightweight, production-ready Retrieval-Augmented Generation (RAG) system designed to analyze system logs (e.g., firewall, router, switch logs) and answer natural language questions about network anomalies or device failures using local LLMs.

## 🔍 Key Features

-  **RAG Pipeline**: Combines FAISS-based semantic search with optional keyword boosting to retrieve relevant log and knowledge base (KB) chunks.
-  **LLM-Based Explanation**: Uses a local open-source language model (`flan-alpaca-base`) to generate root cause and recommendations.
-  **Supports Unstructured Logs**: Parses synthetic syslog-like data into useful answers without needing paid APIs or cloud services.
-  **Firewall/NAT/Anomaly Scenarios**: Includes sample KBs and logs with issues like `FW_DROP_INVALID_PKT`, `NAT_TABLE_OVERFLOW`, etc.
-  **Token-Safe Prompt Handling**: Implements context trimming and chunk scoring to stay within model limits.

## 🛠️ Tech Stack

- **Python**
- **HuggingFace Transformers** – for open LLMs and embeddings
- **SentenceTransformers** – for chunk embedding
- **FAISS** – for vector similarity search
- **Streamlit** – for UI frontend 

## Note
Due to limited data and free model usage, the output might not be accurate.

