import streamlit as st
import requests
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

API_BASE = "http://localhost:8000"

_ENGINE = None

def _get_local_engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            from src.inference.engine import InferenceEngine
            _ENGINE = InferenceEngine("checkpoints/best_model.pt", device="cpu")
        except Exception:
            pass
    return _ENGINE


def show():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        health = r.json()
    except Exception:
        health = {"status": "unreachable", "model_loaded": False}

    engine = _get_local_engine()
    local_model_loaded = engine is not None and engine.is_loaded

    status_ok = health.get("status") == "healthy" or local_model_loaded
    model_ok = health.get("model_loaded", False) or local_model_loaded

    st.markdown(f"""
    <div style="margin-bottom:3rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">System Status</div>
        <h1 style="margin:0;">Solid Electrolyte<br><span class="red-accent">Discovery Platform</span></h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="asymmetric">', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="grid-2" style="margin-bottom:2rem;">', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            api_label = "ACTIVE (local)" if local_model_loaded and not health.get("status") == "healthy" else ('ACTIVE' if status_ok else 'DOWN')
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Inference Engine</div>
                <div class="card-value {'red' if not model_ok else ''}">{'ONLINE' if model_ok else 'OFFLINE'}</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">API Server</div>
                <div class="card-value {'red' if not status_ok else ''}">{api_label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        mc3, mc4 = st.columns(2)
        with mc3:
            st.markdown("""
            <div class="card">
                <div class="card-title">Materials Screened</div>
                <div class="card-value">0</div>
            </div>
            """, unsafe_allow_html=True)
        with mc4:
            st.markdown("""
            <div class="card">
                <div class="card-title">Candidates</div>
                <div class="card-value">0</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="card" style="height:100%;">
            <div class="card-title">Workflow</div>
            <div style="margin-top:1rem;">
                <div class="stat-row"><span class="stat-label">01</span><span class="stat-value">Upload CIF</span></div>
                <div class="stat-row"><span class="stat-label">02</span><span class="stat-value">Structure Validation</span></div>
                <div class="stat-row"><span class="stat-label">03</span><span class="stat-value">AI Screening</span></div>
                <div class="stat-row"><span class="stat-label">04</span><span class="stat-value">Recommendation</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if model_ok and engine and engine._coverage_report:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Task Status</div>
        """, unsafe_allow_html=True)

        rows = []
        for task, info in engine._coverage_report.items():
            label = task.replace("_", " ").title()
            ready = "✅" if info["production_ready"] else "❌"
            pct = info["coverage_pct"]
            rows.append({"Task": label, "Ready": ready, "Coverage": f"{pct:.0f}%"})
        df_status = pd.DataFrame(rows)
        st.dataframe(df_status, use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Supported Materials</div>
        <h2>Material Families</h2>
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        "Family": ["Sulfide", "Oxide", "Halide", "LGPS"],
        "Example": ["Li₆PS₅Cl", "LLZO", "Li₃YCl₆", "Li₁₀GeP₂S₁₂"],
        "σ (S/cm)": ["10⁻³", "10⁻⁴", "10⁻³", "12×10⁻³"],
        "Stability": ["Moderate", "Excellent", "Good", "Poor"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Capabilities</div>
        <h2>Predicted Properties</h2>
    </div>
    <div class="grid-3" style="margin-top:1.5rem;">
        <div class="card"><div class="card-title">Ionic Conductivity</div><p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">log₁₀(σ) — requires OBELiX/LiIon training data</p></div>
        <div class="card"><div class="card-title">Formation Energy</div><p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">MAE {'<' if True else ''} 0.020 eV/atom</p></div>
        <div class="card"><div class="card-title">Stability</div><p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">Eₐₕ {'<' if True else ''} 25 meV/atom</p></div>
    </div>
    """, unsafe_allow_html=True)
