FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && \
    apt-get install -y build-essential curl && \
    apt-get upgrade -y openssl && \
    rm -rf /var/lib/apt/lists/*


# RUN mkdir -p /app/chroma_DB /app/Sqllite/


WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
RUN pip install  -r requirements.txt


EXPOSE 7000


ENV HOST=0.0.0.0
ENV PORT=7000



CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:7000/health || exit 1
