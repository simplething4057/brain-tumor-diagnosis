FROM python:3.11-slim

WORKDIR /app

# System dependencies
# docker.io: brats 패키지가 내부적으로 Docker CLI를 실행하므로 필수
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
