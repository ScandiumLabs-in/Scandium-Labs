#!/bin/bash
cd /home/shamique/Scandium\ Labs\ SSB/scandium-labs
exec venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> api_logs.txt 2>&1
