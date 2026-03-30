FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8601

CMD bash -c "python src/pipelines/rag_build_pipeline.py && streamlit run server.py --server.port 8601 --server.address 0.0.0.0" 