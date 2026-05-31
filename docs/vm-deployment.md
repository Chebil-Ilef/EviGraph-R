# VM Deployment Guide

**VM public IP:** `141.76.56.250`

**Public URLs (once port 80 is open by IT):**
- Frontend UI → `http://141.76.56.250`
- Backend API → `http://141.76.56.250/api/`
- Health check → `http://141.76.56.250/health`

**Internal-only (Docker, not exposed publicly):**
- Frontend container → `localhost:3000`
- Backend container → `localhost:8000`
- Qdrant → `localhost:6333`

---

## How it works (overview)

```
Browser / pip users
       │
       ▼ port 80
  nginx (host)          ← reverse proxy, always running via systemd
  /         /api/
  │              │
  ▼              ▼
port 3000    port 8000
  ui           api        ← Docker containers, restart: always
               │
               ▼
           port 6333
           qdrant          ← Docker container, restart: always
```

There are two separate repos, each with their own Docker Compose file:

| Repo | Path | What it runs |
|------|------|-------------|
| Backend | `/home/service/code/EviGraph-R` | `qdrant` + `api` containers |
| Frontend | `/home/service/code/Evigraph-R-UI` | `ui` (nginx) container |

Two **systemd services** (`evigraph-backend` and `evigraph-frontend`) auto-start both stacks on VM boot. All containers have `restart: always` so they recover from crashes. **nginx** runs as a host service (also systemd, enabled by default) and reverse-proxies port 80 to the containers.

---

## Day-to-day commands

### Check what's running

```bash
# Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
# Expected: evigraph-r-qdrant-1, evigraph-r-api-1, evigraph-r-ui-ui-1

# nginx
sudo systemctl status nginx --no-pager
```

### Check service health

```bash
# Via nginx (port 80 — the public path)
curl http://localhost/health

# Direct container checks
curl http://localhost:8000/health
curl http://localhost:6333/healthz
curl -s http://localhost:3000 | head -3
```

### View logs

```bash
# API logs (live)
cd /home/service/code/EviGraph-R
docker compose logs -f api

# Qdrant logs
docker compose logs -f qdrant

# Frontend logs
cd /home/service/code/Evigraph-R-UI
docker compose logs -f ui

# nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# API debug log file
tail -f /home/service/code/EviGraph-R/logs/debug_logs.txt
```

### Restart a service

```bash
# Restart just the API (e.g. after a code change)
cd /home/service/code/EviGraph-R
docker compose restart api

# Restart entire backend stack
cd /home/service/code/EviGraph-R
docker compose down && docker compose up -d

# Restart frontend
cd /home/service/code/Evigraph-R-UI
docker compose down && docker compose up -d

# Restart nginx
sudo systemctl reload nginx    # graceful (preferred)
sudo systemctl restart nginx   # full restart
```

---

## Deploying code changes

### Backend — Python code change (under `src/`)

`./src` is live-mounted into the container so changes take effect immediately — just restart the api process:

```bash
cd /home/service/code/EviGraph-R
docker compose restart api
```

### Backend — dependency or Dockerfile change

If you change `pyproject.toml`, `uv.lock`, or `Dockerfile`, rebuild the image first:

```bash
cd /home/service/code/EviGraph-R
docker compose build api
docker compose up -d
```

### Frontend change

Vite compiles at build time, so any React source change requires a rebuild:

```bash
cd /home/service/code/Evigraph-R-UI
docker compose build
docker compose up -d
```

---

## nginx reverse proxy

nginx runs directly on the VM host (not in Docker) and routes port 80 traffic to the containers.

**Config file:** `/etc/nginx/sites-available/evigraph`  
**Enabled via symlink:** `/etc/nginx/sites-enabled/evigraph`

Routes:
- `http://141.76.56.250/` → frontend on `localhost:3000`
- `http://141.76.56.250/api/` → backend on `localhost:8000` (SSE streaming enabled)
- `http://141.76.56.250/health` → backend health check

### Edit the nginx config

```bash
sudo nano /etc/nginx/sites-available/evigraph

# After editing, always test before reloading:
sudo nginx -t
sudo systemctl reload nginx
```

### nginx is already enabled on boot

nginx is managed by its own systemd unit (enabled by default on install). You don't need to do anything special — it starts before the Docker containers and waits for connections.

---

## Systemd services (auto-start on reboot)

### Check status

```bash
sudo systemctl status evigraph-backend
sudo systemctl status evigraph-frontend
sudo systemctl status nginx
```

### Manually start / stop

```bash
sudo systemctl start evigraph-backend
sudo systemctl stop evigraph-backend

sudo systemctl start evigraph-frontend
sudo systemctl stop evigraph-frontend
```

### Check if enabled on boot

```bash
sudo systemctl is-enabled evigraph-backend    # should print: enabled
sudo systemctl is-enabled evigraph-frontend   # should print: enabled
sudo systemctl is-enabled nginx               # should print: enabled
```

### View systemd logs (if a service fails to start)

```bash
sudo journalctl -u evigraph-backend --no-pager | tail -40
sudo journalctl -u evigraph-frontend --no-pager | tail -20
```

### Service files

```
/etc/systemd/system/evigraph-backend.service
/etc/systemd/system/evigraph-frontend.service
```

If you edit these files, run `sudo systemctl daemon-reload` afterwards.

---

## What to do after a VM reboot

Nothing — systemd handles everything. To verify after a reboot:

```bash
# Wait ~60 seconds for Qdrant to load its collection, then:
curl http://localhost/health
# Expected: {"status":"ok","qdrant":"ok","collection":"unarxive_chunks","model":"bge-m3"}
```

If it fails, check:

```bash
sudo journalctl -u evigraph-backend --no-pager | tail -30
cd /home/service/code/EviGraph-R && docker compose logs --tail=50
```

---

## Full teardown and restart from scratch

```bash
# Stop everything
sudo systemctl stop evigraph-frontend
sudo systemctl stop evigraph-backend

# Remove containers (Qdrant data in ./storage is safe — it's a volume mount)
cd /home/service/code/EviGraph-R && docker compose down
cd /home/service/code/Evigraph-R-UI && docker compose down

# Start everything back up
sudo systemctl start evigraph-backend
sudo systemctl start evigraph-frontend
```

---

## Ports summary

| Port | Service | Exposed publicly |
|------|---------|-----------------|
| `80` | nginx reverse proxy | Yes (once IT opens it) |
| `3000` | Frontend container | No (internal only) |
| `8000` | Backend API container | No (internal only) |
| `6333` | Qdrant HTTP | No (internal only) |
| `6334` | Qdrant gRPC | No (internal only) |

> The institutional firewall on this VM blocks all inbound traffic except what IT explicitly opens. Port 80 has been requested — until approved, use SSH tunnel (see below).

---

## Accessing before IT opens port 80

Run this on your **local machine**:

```bash
ssh -L 8080:localhost:80 service@141.76.56.250
```

Then open `http://localhost:8080` in your browser — it tunnels through SSH to nginx on the VM.

---

## Troubleshooting

**API container keeps restarting**
```bash
cd /home/service/code/EviGraph-R
docker compose logs api --tail=50
```
Common causes: missing `.env` values (`LLM_API_KEY`, `LLM_API_BASE`), Qdrant not ready yet (wait ~60s after boot).

**Qdrant shows "unhealthy" in `docker ps`**  
Normal for the first ~90 seconds while it loads the `unarxive_chunks` collection. The API starts in parallel and is ready by the time real queries arrive.

**nginx returns 502 Bad Gateway**  
The container it's proxying to isn't running. Check:
```bash
docker ps
curl http://localhost:8000/health   # is the API up?
curl http://localhost:3000          # is the UI up?
```

**Port already in use**
```bash
sudo lsof -i :80
sudo lsof -i :8000
docker ps -a | grep <port>
docker rm -f <container-id>
```
