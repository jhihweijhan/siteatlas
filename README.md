# SiteAtlas

Public codebase for SiteAtlas.

## Structure

- `backend/`: FastAPI service for ingestion, retrieval, and chat APIs.
- `extension/`: Chrome extension UI and page extraction logic.

## Development

- Backend: `cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Tests: `cd backend && uv run pytest -q`

Private planning documents, AI workflows, and local agent configuration live in the separate `siteatlas-dev` repository.
