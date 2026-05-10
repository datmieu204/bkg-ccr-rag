FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN python - <<'PY'
import pathlib
import tomllib

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
deps = data.get("project", {}).get("dependencies", [])
pathlib.Path("requirements.txt").write_text("\n".join(deps))
PY

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--help"]
