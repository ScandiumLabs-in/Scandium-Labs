import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_HOST", "http://localhost:8000")

def auto_refresh(interval=5):
    if st.button("⟳ Refresh", use_container_width=True):
        pass

def show():
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Results</div>
        <h1 style="margin:0;">Job<br><span class="red-accent">Status</span></h1>
    </div>
    """, unsafe_allow_html=True)

    job_id = st.text_input("", placeholder="Paste Job ID here", label_visibility="collapsed")
    st.markdown('<div style="font-size:0.65rem; font-weight:500; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-top:0.25rem;">Job Identifier</div>', unsafe_allow_html=True)

    if not job_id.strip():
        st.markdown("""
        <div style="margin-top:3rem; padding:3rem; border:1px solid #eee; text-align:center;">
            <p style="color:#bbb; font-size:1rem; font-weight:300;">Enter a Job ID to view results</p>
        </div>
        """, unsafe_allow_html=True)
        return

    auto_refresh()

    try:
        r = requests.get(f"{API_BASE}/job/{job_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
        elif r.status_code == 404:
            st.markdown(f"""
            <div style="border:1px solid #ccc; padding:2rem;">
                <div style="font-size:1rem; font-weight:600;">Job not found</div>
                <p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">No job with ID <strong>{job_id}</strong> exists.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        else:
            st.error(f"API returned {r.status_code}")
            return
    except requests.exceptions.ConnectionError:
        st.warning("Backend offline — cannot fetch job status")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    status = data.get("status", "unknown").upper()
    progress = data.get("progress", 0)

    status_color = "#000000"
    if status == "COMPLETED":
        status_color = "#000000"
    elif status in ("FAILED", "ERROR"):
        status_color = "#cc0000"
    elif status in ("PENDING", "QUEUED"):
        status_color = "#999999"

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex; gap:3rem; align-items:center; margin-bottom:2rem;">
        <div>
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999;">Status</div>
            <div style="font-size:1.5rem; font-weight:800; color:{status_color};">{status}</div>
        </div>
        <div>
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999;">Progress</div>
            <div style="font-size:1.5rem; font-weight:800;">{int(progress)}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bar-container" style="margin-bottom:1rem;">
        <div class="bar-fill {'red' if status in ('FAILED','ERROR') else ''}" style="width:{progress:.0f}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    job_name = data.get("job_name", "")
    created = data.get("created_at", "")
    st.markdown(f"""
    <div style="display:flex; gap:2rem; font-size:0.75rem; color:#999;">
        <span><strong>Name:</strong> {job_name or "—"}</span>
        <span><strong>Created:</strong> {created}</span>
    </div>
    """, unsafe_allow_html=True)

    results = data.get("results", data.get("materials", []))
    if results and status == "COMPLETED":
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999;">Output</div>
            <h2 style="margin:0;">Ranked Candidates</h2>
        </div>
        """, unsafe_allow_html=True)

        rows = []
        for r in results:
            mat = r.get("material", r.get("formula", "Unknown"))
            lc = r.get("log_ionic_conductivity", r.get("log_conductivity", None))
            ef = r.get("formation_energy", None)
            eah = r.get("energy_above_hull", None)
            ea = r.get("activation_energy", None)
            rank = r.get("rank", None)

            if lc is not None:
                try:
                    lc = float(lc)
                except Exception:
                    pass
            if ef is not None:
                try:
                    ef = float(ef)
                except Exception:
                    pass
            if eah is not None:
                try:
                    eah = float(eah)
                except Exception:
                    pass
            if ea is not None:
                try:
                    ea = float(ea)
                except Exception:
                    pass

            rows.append({
                "Rank": rank if rank is not None else "—",
                "Material": mat,
                "log σ (S/cm)": f"{lc:.2f}" if isinstance(lc, float) else str(lc),
                "σ (S/cm)": f"{10**lc:.2e}" if isinstance(lc, float) else "—",
                "E₍ (eV/atom)": f"{ef:.3f}" if isinstance(ef, float) else str(ef),
                "Eₐₕ (eV/atom)": f"{eah:.3f}" if isinstance(eah, float) else str(eah),
            })

        df = pd.DataFrame(rows)

        if "Rank" in df.columns:
            df = df.sort_values("Rank" if df["Rank"].dtype in (int, float) else "Material").reset_index(drop=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        chart_rows = []
        for r in results:
            mat = r.get("material", r.get("formula", "Unknown"))
            lc = r.get("log_ionic_conductivity", r.get("log_conductivity"))
            if lc is not None:
                try:
                    lc = float(lc)
                except Exception:
                    continue
                rank = r.get("rank", None)
                if rank is None:
                    rank = 0
                chart_rows.append({"material": str(mat), "log_σ": lc})

        if len(chart_rows) > 1:
            import altair as alt
            chart_df = pd.DataFrame(chart_rows).sort_values("log_σ", ascending=False).reset_index(drop=True)

            bars = alt.Chart(chart_df).mark_bar(
                color="#000000",
                size=20,
            ).encode(
                x=alt.X("material:N", sort=None, title="", axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("log_σ:Q", title="log₁₀ σ (S/cm)", axis=alt.Axis(grid=True, tickSize=0, labelFontSize=10, labelColor="#666")),
                tooltip=["material", "log_σ"],
            ).properties(
                height=280,
                padding={"left": 0, "right": 0, "top": 10, "bottom": 10},
            ).configure_view(
                strokeWidth=0,
            ).configure_axis(
                gridColor="#eeeeee",
                labelFont="Inter, sans-serif",
                titleFont="Inter, sans-serif",
                titleFontSize=11,
                titleFontWeight=600,
                titleColor="#999",
            )

            top = chart_df.head(1)
            highlight = alt.Chart(top).mark_bar(
                color="#cc0000",
                size=20,
            ).encode(
                x=alt.X("material:N", sort=None),
                y=alt.Y("log_σ:Q"),
            )

            st.altair_chart(bars + highlight, use_container_width=True)
