FROM python:3.11-slim

WORKDIR /app


COPY pyproject.toml .


RUN pip install --no-cache-dir .

COPY models/ ./models/
COPY src/ ./src/
COPY app/ ./app/


RUN pip install --no-cache-dir --no-deps .
CMD [ "uvicorn","app.main:app","--host", "0.0.0.0", "--port", "8000" ]
EXPOSE 8000
