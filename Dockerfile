FROM python:3.13-slim

WORKDIR /app

# uv
RUN pip install uv

# dependency management files
COPY pyproject.toml uv.lock ./

# dependencies
RUN uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]