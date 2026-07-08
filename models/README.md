# Production Model

Copy a trained checkpoint here for Docker Compose:

```bash
cp ../checkpoints/best_model.pt best_model.pt
```

This directory is mounted into the `api` and `worker` containers
at `/models/best_model.pt` (set via `MODEL_PATH` in docker-compose.yml).
