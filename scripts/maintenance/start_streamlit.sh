#!/bin/bash
cd "/home/shamique/Scandium Labs SSB/scandium-labs"
exec venv/bin/streamlit run streamlit_app/streamlit_app.py --server.port 8501 --server.headless true >> streamlit_logs.txt 2>&1
