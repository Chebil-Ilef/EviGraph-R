# Plan: EviGraph-R — pip package + VM deployment

## Context

Two parallel goals:
1. **pip package** — `pip install evigraph-r` gives users `import evigraph` and an `evigraph` CLI. Nobody will download 700 GB of arXiv data locally, so the package connects to a **hosted Qdrant instance on this VM** by default. Users can override the URL.
2. **VM deployment** — Both the backend (`EviGraph-R`) and the frontend (`Evigraph-R-UI`) must run permanently on this VM (IP: `141.76.56.250`) via Docker Compose, with a `PROD=true` env flag that switches the Qdrant default from `localhost` to the VM's public address.

The VM already has both repos:
- `/home/service/code/EviGraph-R` — FastAPI backend
- `/home/service/code/Evigraph-R-UI` — React + Nginx frontend

---

## Part A — pip package

### A1. Restructure source into `src/evigraph/`

Currently `src/` is the naked PYTHONPATH root with bare package dirs (`api/`, `agents/`, etc.). For a proper installable package, wrap them in a single namespace:

```
src/
└── evigraph/
    ├── __init__.py      ← public re-exports
    ├── cli.py           ← new
    ├── api/
    ├── agents/
    ├── config/
    ├── retrieval/
    ├── schemas/
    ├── utils/
    ├── visualization/
    ├── workflow/
    └── indexing/
```

`PYTHONPATH=/app/src` in Docker/dev still works — `evigraph` becomes a real package found inside `src/`.

### A2. `pyproject.toml` changes

```toml
[project]
name = "evigraph-r"          # PyPI distribution name
version = "0.1.0"
description = "Evidence-graph reasoning over 2.8M arXiv papers"
authors = [{name = "EviGraph Team"}]
license = {text = "MIT"}
requires-python = ">=3.11"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/your-org/EviGraph-R"

[project.scripts]
evigraph = "evigraph.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"
```

### A3. Internal import rewriting

Every bare import becomes prefixed with `evigraph.`:

```python
# before
from agents.decomposer import DecomposerAgent
from utils.llm import get_llm_client
from config.settings import PATHS

# after
from evigraph.agents.decomposer import DecomposerAgent
from evigraph.utils.llm import get_llm_client
from evigraph.config.settings import PATHS
```

Affected modules: `api/`, `agents/`, `workflow/`, `retrieval/`, `utils/`, `indexing/`, `visualization/`, `config/`.  
Strategy: `find src/ -name "*.py" | xargs sed -i` with targeted patterns.

### A4. Fix `PROJECT_ROOT` in `settings.py`

After moving to `src/evigraph/config/settings.py` (one level deeper):

```python
# before (src/config/settings.py — 3 parents: config → src → repo_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# after (src/evigraph/config/settings.py — 4 parents: config → evigraph → src → repo_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
```

### A5. `src/evigraph/__init__.py` — public API

```python
from evigraph.api.runner import WorkflowRunner
from evigraph.api.schemas import QueryRequest, QueryResponse, PipelineConfig
from evigraph.schemas.objects import EvidenceGraph, AnnotatedSentence, SubQuery

__all__ = ["WorkflowRunner", "QueryRequest", "QueryResponse",
           "PipelineConfig", "EvidenceGraph", "AnnotatedSentence", "SubQuery"]
```

### A6. `src/evigraph/cli.py` — CLI entry point

Commands:
- `evigraph query "<question>"` — runs `WorkflowRunner.run_query()`, prints answer
- `evigraph serve` — launches FastAPI via `uvicorn.run()`
- `evigraph --version`

Flags:
- `--qdrant-url` on both `query` and `serve` — overrides `QDRANT_URL`
- `--json` on `query` — full JSON output
- `--model`, `--top-k` on `query`
- `--host`, `--port`, `--reload` on `serve`

```python
import argparse, asyncio, os
from importlib.metadata import version

def _cmd_query(args):
    if args.qdrant_url:
        os.environ["QDRANT_URL"] = args.qdrant_url
    from evigraph.api.runner import WorkflowRunner
    from evigraph.api.schemas import QueryRequest, PipelineConfig
    runner = WorkflowRunner(model_key=args.model)
    request = QueryRequest(query=args.question, config=PipelineConfig(top_k=args.top_k))
    result = asyncio.run(runner.run_query(request))
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.answer)

def _cmd_serve(args):
    if args.qdrant_url:
        os.environ["QDRANT_URL"] = args.qdrant_url
    import uvicorn
    uvicorn.run("evigraph.api.main:app", host=args.host, port=args.port,
                reload=args.reload, workers=1)

def main():
    parser = argparse.ArgumentParser(prog="evigraph")
    parser.add_argument("--version", action="version", version=version("evigraph-r"))
    sub = parser.add_subparsers(dest="command", required=True)

    qp = sub.add_parser("query")
    qp.add_argument("question")
    qp.add_argument("--model", default="bge-m3")
    qp.add_argument("--top-k", type=int, default=15)
    qp.add_argument("--qdrant-url", default=None)
    qp.add_argument("--json", action="store_true")
    qp.set_defaults(func=_cmd_query)

    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.add_argument("--qdrant-url", default=None)
    sp.set_defaults(func=_cmd_serve)

    args = parser.parse_args()
    args.func(args)
```

### A7. Hosted Qdrant default + `PROD` flag

In `src/evigraph/config/settings.py`, the `_QdrantConnection.url` default changes:

```python
# current
url: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_URL"))

# new
_PROD = os.getenv("PROD", "false").lower() in ("1", "true", "yes")
_DEFAULT_QDRANT_URL = "http://141.76.56.250:6333" if _PROD else None

url: Optional[str] = field(
    default_factory=lambda: os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL)
)
```

- `PROD=false` (dev/Docker): `QDRANT_URL` env var wins; if unset → `None` → connects to `localhost:6333` (Docker sidecar). Existing Docker Compose unaffected.
- `PROD=true` (pip users, VM): defaults to `http://141.76.56.250:6333`.
- Any user can always set `QDRANT_URL=http://...` to override.

Add a friendly startup error in `src/evigraph/utils/qdrant.py` if the Qdrant host is unreachable:

```python
# at the top of qdrant_client()
try:
    client = QdrantClient(...)
    client.get_collections()   # cheap health check
except Exception as e:
    raise ConnectionError(
        f"Cannot reach Qdrant at {connection.url or connection.host}:{connection.port}. "
        "Set QDRANT_URL to a reachable EviGraph-R Qdrant instance, or run the full "
        "stack with `docker compose up`."
    ) from e
```

### A8. Update root `main.py`

```python
from evigraph.cli import main
if __name__ == "__main__":
    main()
```

---

## Part B — VM deployment (always-running)

### B1. Single production Compose file

Create `/home/service/docker-compose.prod.yml` combining backend + frontend + qdrant with `restart: always`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.13.6
    restart: always
    ports:
      - "6333:6333"
    ulimits:
      nofile: {soft: 524288, hard: 524288}
    volumes:
      - /home/service/code/EviGraph-R/storage:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  api:
    build: /home/service/code/EviGraph-R
    restart: always
    ports:
      - "8000:8000"
    env_file: /home/service/code/EviGraph-R/.env
    environment:
      INDEXING_PROFILE: local
      QDRANT_URL: http://qdrant:6333
      PYTHONPATH: /app/src
      HF_HOME: /app/hub
      PROD: "false"    # uses compose-internal qdrant, not public IP
    volumes:
      - /home/service/code/EviGraph-R/hub:/app/hub
      - /home/service/code/EviGraph-R/src:/app/src
      - /home/service/code/EviGraph-R/logs:/app/logs
    depends_on:
      qdrant:
        condition: service_healthy

  ui:
    build: /home/service/code/Evigraph-R-UI
    restart: always
    ports:
      - "3000:3000"
    depends_on:
      - api
```

**Note on nginx proxy:** The frontend nginx proxies `/api/` to `http://172.17.0.1:8000`. On this VM, `172.17.0.1` is the Docker bridge host IP, which routes to port 8000 bound by the `api` container. No nginx change needed.

### B2. Firewall — open required ports

```bash
sudo ufw allow 3000/tcp   # frontend
sudo ufw allow 8000/tcp   # backend API (pip users + nginx proxy)
sudo ufw allow 6333/tcp   # Qdrant (pip users hit the hosted index)
sudo ufw reload
```

### B3. Systemd service — auto-start on reboot

`/etc/systemd/system/evigraph.service`:

```ini
[Unit]
Description=EviGraph-R full stack (qdrant + api + ui)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/service
ExecStart=/usr/bin/docker compose -f /home/service/docker-compose.prod.yml up -d --build
ExecStop=/usr/bin/docker compose -f /home/service/docker-compose.prod.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable evigraph
sudo systemctl start evigraph
```

---

## Execution order

### Phase 1 — pip package
1. Create `src/evigraph/` and move all subdirs into it
2. Rewrite all internal imports (`from agents.*` → `from evigraph.agents.*`)
3. Fix `PROJECT_ROOT` depth in `settings.py`
4. Add `PROD` flag + hosted Qdrant default to `settings.py`
5. Add startup `ConnectionError` to `utils/qdrant.py`
6. Create `src/evigraph/__init__.py`
7. Create `src/evigraph/cli.py`
8. Update `pyproject.toml`
9. Update root `main.py`
10. Smoke-test: `pip install -e . && evigraph --version && python -c "import evigraph"`

### Phase 2 — VM deployment
11. Create `/home/service/docker-compose.prod.yml`
12. Check/open firewall ports
13. Create `/etc/systemd/system/evigraph.service` and enable it
14. Start stack: `sudo systemctl start evigraph`
15. Verify: `curl http://141.76.56.250:8000/health` and open `http://141.76.56.250:3000` in browser

---

## Files changed

| File | Action |
|------|--------|
| `src/evigraph/` | New directory (subdirs moved from bare `src/`) |
| `src/evigraph/__init__.py` | New — public API |
| `src/evigraph/cli.py` | New — CLI |
| `src/evigraph/config/settings.py` | Fix PROJECT_ROOT depth; add PROD flag + hosted Qdrant default |
| `src/evigraph/utils/qdrant.py` | Add friendly ConnectionError on unreachable Qdrant |
| All `src/**/*.py` | Rewrite bare imports → `evigraph.*` prefix |
| `pyproject.toml` | build-system, package discovery, scripts, PyPI metadata |
| `main.py` (repo root) | Delegate to `evigraph.cli:main` |
| `/home/service/docker-compose.prod.yml` | New — production compose (qdrant + api + ui) |
| `/etc/systemd/system/evigraph.service` | New — auto-start on reboot |
| `Dockerfile`, `docker-compose.yml`, `hpc_serve.sh`, `uv.lock` | **Unchanged** |

---

## Verification

```bash
# pip package
pip install -e .
evigraph --version
PROD=true evigraph query "What is attention in transformers?" --json

# VM stack
curl http://141.76.56.250:8000/health
# Browser: http://141.76.56.250:3000

# After VM reboot
sudo reboot
# wait ~60s
curl http://141.76.56.250:8000/health
```
