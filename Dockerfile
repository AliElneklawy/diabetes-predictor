FROM ghcr.io/astral-sh/uv:0.8.22-debian

WORKDIR /app
COPY . .

RUN uv venv \
    && uv pip install .

EXPOSE 80

CMD ["uv", "run", "src/app.py"]
