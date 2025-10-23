FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY docs_dir/ docs_dir/
COPY start.sh ./start.sh


RUN chmod +x start.sh

EXPOSE 8000 8501

ENV DOCUMENTS_DIR="/app/docs_dir"
ENV VECTOR_STORE_DIR="/app/doc_vector_store"
ENV COLLECTION_NAME="document_collection"
ENV MODEL_NAME="llama-3.3-70b-versatile"
ENV MODEL_TEMPERATURE=0.0
ENV CHAT_ENDPOINT_URL="http://localhost:8000/chat/answer"

CMD ["/app/start.sh"]
