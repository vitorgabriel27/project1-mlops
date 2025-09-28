FROM python:3.10-slim

# 1. Defina diretório de trabalho
WORKDIR /app

# 2. Instale dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Instale o Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# 4. Copie apenas arquivos de dependência primeiro para aproveitar cache do Docker
COPY pyproject.toml poetry.lock* ./

# 5. Instale as dependências com Poetry (sem venv) + torch via pip
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi && \
    pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 6. Copie o restante do código
COPY . .

# 7. Exponha a porta do Streamlit
EXPOSE 8501

# 8. Verificação de saúde
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. EntryPoint para rodar o Streamlit com poetry
ENTRYPOINT ["poetry", "run", "streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

