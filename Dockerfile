FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt --no-cache-dir
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY data/ /app/data/
EXPOSE 8000
CMD ["pytest", "tests/"]