# AGNT5 Workerless with FastAPI

This example mounts the AGNT5 workerless protocol into a normal FastAPI
service. Your existing service routes stay in the same app, while AGNT5 uses:

- `GET /.well-known/agnt5`
- `POST /agnt5/invoke`

## Local Run

From this directory:

```bash
export AGNT5_SERVERLESS_SIGNING_SECRET="$(openssl rand -base64 32)"
uv run --with-editable ../.. --with fastapi --with "uvicorn[standard]" \
  uvicorn app:app --host 127.0.0.1 --port 8787
```

Validate the existing service route:

```bash
curl http://127.0.0.1:8787/health
```

Validate the AGNT5 manifest:

```bash
curl http://127.0.0.1:8787/.well-known/agnt5
```

## FastAPI Mounting Pattern

```python
from fastapi import FastAPI
from agnt5.serverless import serve

app = FastAPI()
agnt5_workerless = serve(...)
agnt5_workerless.mount_fastapi(app)
```

Use the same deployed HTTPS service URL with `agnt5 serverless sync`:

```bash
agnt5 serverless sync \
  https://<fastapi-host> \
  --provider node \
  --env production \
  --immutable-ref <git-sha-or-release-id> \
  --signing-secret-env AGNT5_SERVERLESS_SIGNING_SECRET
```

Use `--provider node` until AGNT5 adds a dedicated Python/FastAPI provider
label. The runtime protocol is the same workerless HTTP protocol used by the
TypeScript serverless adapters.
