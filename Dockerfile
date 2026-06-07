FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8501
EXPOSE 8501

# Deployment marker: rollback build after removing child-level batch refactor.
# Railway sets $PORT. We bind to 0.0.0.0 so the service is reachable.
CMD ["sh", "-c", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true"]
