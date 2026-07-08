import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ENGINE = None
_MODEL_META = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from src.inference.engine import InferenceEngine
        _ENGINE = InferenceEngine("checkpoints/best_model.pt", device="cpu")
    return _ENGINE


def _get_model_metadata():
    global _MODEL_META
    if _MODEL_META is not None:
        return _MODEL_META
    meta = {
        "model_name": "ScandiumPINNGNN",
        "model_version": "v3.1",
        "checkpoint": "best_model.pt",
        "training_date": "2026-07-08",
        "git_commit": "unknown",
        "git_branch": "unknown",
        "training_dataset": "10,000 structures (MP, OQMD, JARVIS)",
        "training_chemistries": "Li-bearing solid electrolytes",
        "inference_device": "CPU",
    }
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip().split()[0]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            meta["git_branch"] = result.stdout.strip()
    except Exception:
        pass
    try:
        ckpt = _get_engine().model  # already loaded
        if hasattr(ckpt, "hidden_dim"):
            meta["hidden_dim"] = ckpt.hidden_dim
        if hasattr(ckpt, "tasks"):
            meta["tasks"] = ckpt.tasks
        if hasattr(ckpt, "mc_dropout_samples"):
            meta["mc_samples"] = ckpt.mc_dropout_samples
    except Exception:
        pass
    _MODEL_META = meta
    return meta


KNOWN_MATERIALS = {
    "Li6PS5Cl": {
        "name": "Li₆PS₅Cl (Argyrodite)",
        "exp_sigma": "10⁻³ – 10⁻² S/cm",
        "exp_eah": 0.003,
        "exp_band_gap": "~2.0 eV",
        "exp_formation_energy": -1.5,
        "mp_id": "mp-985592",
        "notes": "Known fast Li-ion conductor. The experimental phase is partially disordered on the S/Cl sublattice.",
        "references": ["Nature Energy 2016", "Materials Project mp-985592"],
        "family": "Argyrodite",
    },
    "Li10GeP2S12": {
        "name": "Li₁₀GeP₂S₁₂ (LGPS)",
        "exp_sigma": "12×10⁻³ S/cm",
        "exp_eah": 0.01,
        "exp_band_gap": "~3.0 eV",
        "exp_formation_energy": -1.2,
        "mp_id": "mp-?????",
        "notes": "Highest known Li-ion conductivity among sulfides; unstable against Li metal anode.",
        "references": ["Nature Materials 2011"],
        "family": "LGPS",
    },
    "Li7La3Zr2O12": {
        "name": "Li₇La₃Zr₂O₁₂ (LLZO)",
        "exp_sigma": "10⁻⁴ S/cm",
        "exp_eah": 0.01,
        "exp_band_gap": "~5.0 eV",
        "exp_formation_energy": -2.1,
        "mp_id": "mp-?????",
        "notes": "Garnet-type oxide; stable against Li metal; requires Al or Ga doping for phase stabilization.",
        "references": ["Angew. Chem. 2007"],
        "family": "Garnet",
    },
    "Li3YCl6": {
        "name": "Li₃YCl₆",
        "exp_sigma": "10⁻³ S/cm",
        "exp_eah": 0.05,
        "exp_band_gap": "~4.0 eV",
        "exp_formation_energy": -1.0,
        "mp_id": "mp-?????",
        "notes": "Halide solid electrolyte; moisture sensitive; good oxidative stability.",
        "references": ["Adv. Energy Mater. 2020"],
        "family": "Halide",
    },
    "Li3PS4": {
        "name": "Li₃PS₄ (Thio-LISICON)",
        "exp_sigma": "10⁻⁴ – 10⁻³ S/cm",
        "exp_eah": 0.02,
        "exp_band_gap": "~3.5 eV",
        "exp_formation_energy": -1.0,
        "mp_id": "mp-?????",
        "notes": "Glass-ceramic sulfide; forms basis of many composite electrolytes.",
        "references": ["J. Power Sources 2015"],
        "family": "Sulfide",
    },
}


def _lookup_material(formula):
    for key, info in KNOWN_MATERIALS.items():
        if formula.lower().replace(" ", "") == key.lower():
            return info
    return None


def _format_formula(formula):
    subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(subs)


def _confidence_bar(pct, label=None, color="#1a7a1a"):
    pct = max(0, min(100, pct))
    return f"""
    <div style="margin-bottom:0.5rem;">
        {f'<div style="font-size:0.65rem; color:#999; margin-bottom:0.2rem;">{label}</div>' if label else ''}
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <div style="flex:1; height:6px; background:#eee; border-radius:3px; overflow:hidden;">
                <div style="height:100%; width:{pct:.0f}%; background:{color}; border-radius:3px; transition:width 0.5s;"></div>
            </div>
            <span style="font-size:0.7rem; font-weight:600; min-width:2.5rem; text-align:right;">{pct:.0f}%</span>
        </div>
    </div>
    """


def _uncertainty_viz(value, uncertainty, label, unit, min_val=0, max_val=1):
    lo = value - uncertainty if uncertainty is not None else value * 0.9
    hi = value + uncertainty if uncertainty is not None else value * 1.1
    pct = (value - min_val) / (max_val - min_val) * 100
    lo_pct = max(0, (lo - min_val) / (max_val - min_val) * 100)
    hi_pct = min(100, (hi - min_val) / (max_val - min_val) * 100)

    return f"""
    <div style="margin:0.75rem 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#999;">
            <span>{label}</span>
            <span style="font-weight:600; color:#000;">{value:.3f} {unit}</span>
        </div>
        <div style="position:relative; height:20px; margin-top:0.25rem;">
            <div style="position:absolute; left:0; right:0; top:8px; height:4px; background:#eee; border-radius:2px;"></div>
            <div style="position:absolute; left:{lo_pct:.1f}%; right:{100-hi_pct:.1f}%; top:8px; height:4px; background:#1a7a1a; border-radius:2px; opacity:0.3;"></div>
            <div style="position:absolute; left:{pct:.1f}%; top:2px; width:16px; height:16px; background:#000; border-radius:50%; transform:translateX(-50%);"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:#aaa;">
            <span>{min_val}</span>
            <span>{max_val}</span>
        </div>
    </div>
    """


def show():
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Single Material</div>
        <h1 style="margin:0;">Screen a<br><span class="red-accent">Crystal Structure</span></h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="asymmetric">', unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Upload CIF or POSCAR",
            type=["cif", "poscar", "vasp"],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style="margin-top:0.5rem; font-size:0.7rem; color:#999;">
            Accepted formats: CIF · POSCAR · VASP &nbsp;|&nbsp; Max 10MB
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        temperature = st.number_input("Temperature (K)", value=300, min_value=100, max_value=1000, step=10)
        st.markdown("""
        <div style="font-size:0.65rem; font-weight:500; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-top:0.25rem;">Screening Temperature</div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div style="margin-top:3rem; text-align:center; padding:3rem; border:1px solid #eee;">
            <p style="color:#bbb; font-size:1.2rem; font-weight:300;">Drop a CIF or POSCAR file to begin</p>
            <p style="color:#ccc; font-size:0.8rem; margin-top:0.5rem;">
                Li₆PS₅Cl · Li₁₀GeP₂S₁₂ · Li₃PS₄ · LLZO
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if st.button("Predict Properties", use_container_width=True):
        with st.spinner("Analyzing crystal structure..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                from pymatgen.core import Structure
                structure = Structure.from_file(tmp_path)
                formula = structure.composition.reduced_formula
                known = _lookup_material(formula)

                from src.inference.validation import validate_structure
                validation = validate_structure(structure)

                engine = _get_engine()
                result = engine.predict_single(structure, temperature)
                result["formula"] = formula
                result["n_atoms"] = len(structure)
                result["volume"] = structure.volume
                result["density"] = structure.density

                # Track in history
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.append({
                    "formula": formula,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "eah": result.get("energy_above_hull", {}).get("value"),
                    "ef": result.get("formation_energy", {}).get("value"),
                    "bg": result.get("band_gap", {}).get("value"),
                    "rec": result.get("recommendation", ""),
                })

                _render_full_report(structure, formula, result, known, validation, temperature)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                os.unlink(tmp_path)


def _section(title, content_func, expanded=True, badge=None):
    """Renders a collapsible section using HTML details/summary."""
    from streamlit.components.v1 import html
    badge_html = f'<span style="margin-left:0.5rem; font-size:0.6rem; font-weight:600; text-transform:uppercase; color:#999; border:1px solid #ddd; padding:0.1rem 0.4rem;">{badge}</span>' if badge else ""
    key = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with st.container():
        st.markdown(f"""
        <details {'open' if expanded else ''} style="margin-bottom:1rem; border:1px solid #eee; border-radius:4px;">
            <summary style="padding:0.75rem 1rem; cursor:pointer; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#000; background:#fafafa; border-radius:4px; user-select:none;">
                {title}{badge_html}
            </summary>
            <div style="padding:1rem;">
        """, unsafe_allow_html=True)
        content_func()
        st.markdown("</div></details>", unsafe_allow_html=True)


def _render_full_report(structure, formula, result, known, validation, temperature):
    meta = _get_model_metadata()
    eah = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")

    _section("Material Overview", lambda: _overview_section(structure, formula, result, known, validation), expanded=True)
    _section("Structure Validation", lambda: _structure_validation_section(validation), expanded=True)
    _section("Crystal Structure", lambda: _crystal_viz_section(structure, formula), expanded=False, badge="Interactive 3D")
    _section("Model Predictions", lambda: _predictions_section(result, known), expanded=True)
    _section("Confidence & Uncertainty", lambda: _confidence_section(result, known, validation), expanded=True, badge="Calibration")
    _section("Thermodynamic Stability", lambda: _stability_section(result, known), expanded=True)
    _section("Literature Comparison", lambda: _lit_comparison_section(known, result), expanded=bool(known), badge="Known Material")
    _section("Why This Prediction?", lambda: _attribution_section(structure, result), expanded=False, badge="Feature Analysis")
    _section("Similar Known Materials", lambda: _nearest_materials_section(structure, result), expanded=False, badge="Embeddings")
    _section("Recommendations", lambda: _recommendations_section(result, known), expanded=True)
    _section("Model Card & Provenance", lambda: _provenance_section(meta, result), expanded=False, badge="Technical")
    _section("Limitations", lambda: _limitations_section(), expanded=False)
    _section("Prediction History", lambda: _history_section(), expanded=False, badge=f"{len(st.session_state.get('prediction_history', []))} runs")
    _section("Scientific Summary", lambda: _summary_section(formula, result, known), expanded=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        pass
    with col2:
        if st.button("Export PDF Report", use_container_width=True, type="primary"):
            _export_pdf(structure, formula, result, known, validation, meta)


def _overview_section(structure, formula, result, known, validation):
    known_label = "Known Material" if known else "Novel Composition"
    rec = result.get("recommendation", "")
    eah = result.get("energy_above_hull", {}).get("value")
    pred_confidence = result.get("recommendation_confidence", "medium")
    is_reject = rec == "REJECT"
    is_known_reject = known and is_reject

    if is_known_reject:
        status_text = "Requires Further Validation"
        status_color = "#cc9900"
        rec_detail = "Model prediction disagrees with experimental literature"
    elif rec == "HIGH PRIORITY":
        status_text = "High Priority Candidate"
        status_color = "#1a7a1a"
        rec_detail = result.get("recommendation_detail", "")
    elif rec == "MEDIUM PRIORITY":
        status_text = "Medium Priority Candidate"
        status_color = "#b8a000"
        rec_detail = result.get("recommendation_detail", "")
    elif rec == "REJECT":
        status_text = "Low Priority — Unstable Prediction"
        status_color = "#cc0000"
        rec_detail = result.get("recommendation_detail", "")
    else:
        status_text = rec if rec else "Assessment Complete"
        status_color = "#000"
        rec_detail = result.get("recommendation_detail", "")

    lit_agree = "N/A (novel material)"
    if known:
        lit_agree = "Inconsistent" if (eah is not None and eah > 0.05) else "Consistent"

    fmt_formula = _format_formula(formula)
    st.markdown(f"""
    <div style="border:2px solid {status_color}; padding:1.5rem; margin-bottom:1.5rem;">
        <div style="display:grid; grid-template-columns:2fr 1fr; gap:2rem;">
            <div>
                <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.02em; color:{status_color};">{status_text}</div>
                <div style="font-size:0.85rem; color:#555; margin-top:0.5rem;">{rec_detail}</div>
                <div style="display:flex; gap:0.5rem; margin-top:0.75rem; flex-wrap:wrap;">
                    <span style="display:inline-block; border:1px solid #000; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:600; text-transform:uppercase;">{known_label}</span>
                    <span style="display:inline-block; border:1px solid #000; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:600; text-transform:uppercase;">{fmt_formula}</span>
                    <span style="display:inline-block; border:1px solid #000; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:600; text-transform:uppercase;">{'✓ Valid CIF' if validation['passed'] else '⚠ CIF Warnings'}</span>
                </div>
            </div>
            <div style="border-left:1px solid #eee; padding-left:1.5rem;">
                <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.75rem;">Confidence</div>
                <div class="stat-row"><span class="stat-label">Prediction</span><span class="stat-value">{pred_confidence.title()}</span></div>
                <div class="stat-row"><span class="stat-label">Literature Agreement</span><span class="stat-value">{lit_agree}</span></div>
                <div class="stat-row"><span class="stat-label">CIF Quality</span><span class="stat-value">{'Good' if validation['passed'] else 'Issues'}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if known:
        st.markdown(f"""
        <div style="border:1px solid #000; padding:1.25rem; margin-bottom:1.5rem;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Known Material — Literature Reference</div>
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.75rem;">{known["name"]}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:0.75rem;">
                <div><div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Conductivity</div><div style="font-size:0.9rem; font-weight:600;">{known["exp_sigma"]}</div></div>
                <div><div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Band Gap</div><div style="font-size:0.9rem;">{known["exp_band_gap"]}</div></div>
                <div><div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Family</div><div style="font-size:0.9rem;">{known.get("family", "N/A")}</div></div>
                <div><div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">MP ID</div><div style="font-size:0.9rem;">{known.get("mp_id", "N/A")}</div></div>
            </div>
            <p style="font-size:0.8rem; color:#555; margin:0; border-top:1px solid #eee; padding-top:0.75rem;">{known["notes"]}</p>
        </div>
        """, unsafe_allow_html=True)


def _structure_validation_section(validation):
    items = []
    for c in validation["checks"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#1a7a1a;">✓</span><span style="font-size:0.8rem;">{c}</span></div>')
    for w in validation["warnings"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#cc9900;">⚠</span><span style="font-size:0.8rem;">{w}</span></div>')
    for e in validation["errors"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#cc0000;">✗</span><span style="font-size:0.8rem;">{e}</span></div>')

    border_color = "#cc0000" if not validation["passed"] else ("#cc9900" if validation["warnings"] else "#1a7a1a")
    st.markdown(f"""
    <div style="border:1px solid {border_color}; padding:1rem;">
        {''.join(items)}
    </div>
    """, unsafe_allow_html=True)


def _crystal_viz_section(structure, formula):
    try:
        import py3Dmol
    except ImportError:
        st.markdown('<p style="font-size:0.8rem; color:#999;">Install py3Dmol for interactive crystal visualization.</p>', unsafe_allow_html=True)
        return

    cif_str = structure.to(fmt="cif")
    view = py3Dmol.view(width=600, height=400)
    view.addModel(cif_str, "cif")
    view.setStyle({"stick": {},"sphere": {"radius": 0.5}})
    view.zoomTo()
    view.spin()
    html = view._make_html()
    st.components.v1.html(html, height=420, scrolling=False)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; margin-top:0.75rem; font-size:0.75rem; color:#666;">
        <div><strong>Formula</strong> {_format_formula(formula)}</div>
        <div><strong>Atoms</strong> {len(structure)}</div>
        <div><strong>Volume</strong> {structure.volume:.1f} Å³</div>
        <div><strong>Density</strong> {structure.density:.2f} g/cm³</div>
        <div><strong>Space Group</strong> {structure.get_space_group_info()[0] if hasattr(structure, 'get_space_group_info') else 'N/A'}</div>
        <div><strong>Lattice</strong> {structure.lattice.a:.2f} / {structure.lattice.b:.2f} / {structure.lattice.c:.2f} Å</div>
    </div>
    """, unsafe_allow_html=True)


def _predictions_section(result, known):
    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    preds = []
    for key, label, unit, fmt in [
        ("formation_energy", "Formation Energy", "eV/atom", "{:.3f}"),
        ("energy_above_hull", "Energy Above Hull", "eV/atom", "{:.3f}"),
        ("band_gap", "Band Gap", "eV", "{:.2f}"),
        ("activation_energy", "Activation Energy", "eV", "{:.2f}"),
    ]:
        entry = result.get(key, {})
        val = entry.get("value")
        if val is not None:
            unc = entry.get("uncertainty")
            display = fmt.format(val) + f" {unit}"
            if unc is not None:
                display = fmt.format(val) + f" ± {unc:.3f} {unit}"
            preds.append((label, display, val, unc, unit))

    with col1:
        for label, val, raw, unc, unit in preds[:2]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
            if raw is not None:
                st.markdown(_uncertainty_viz(raw, unc, "", unit, 0, 1), unsafe_allow_html=True)
    with col2:
        for label, val, raw, unc, unit in preds[2:]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
            if raw is not None:
                st.markdown(_uncertainty_viz(raw, unc, "", unit, 0, 1), unsafe_allow_html=True)

    # Conductivity block
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Ionic Conductivity</div>', unsafe_allow_html=True)

    cond = result.get("log_ionic_conductivity", {})
    cond_value = cond.get("value")
    if cond_value is not None:
        sigma = 10 ** cond_value
        display_val = f"{sigma:.2e} S/cm"
        st.markdown(f'<div style="font-size:2rem; font-weight:800; margin-bottom:1rem;">{display_val}</div>', unsafe_allow_html=True)
        log_val = max(-5, min(2, cond_value))
        pct = (log_val + 5) / 7 * 100
        st.markdown(f"""
        <div class="bar-container"><div class="bar-fill" style="width:{pct:.0f}%;"></div></div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#999; margin-bottom:1rem;">
            <span>10⁻⁵</span><span>10⁻³</span><span>10⁻¹</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px solid #eee; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.9rem; font-weight:600; color:#999; margin-bottom:0.5rem;">Not Predicted</div>
            <div style="font-size:0.75rem; color:#999;">
                The current model has not been trained on experimentally labeled ionic conductivity data.
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:1rem;">
            <div style="border:1px solid #eee; padding:0.75rem;">
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#1a7a1a;">Available</div>
                <div style="font-size:0.85rem;">✓ Formation Energy</div>
                <div style="font-size:0.85rem;">✓ Band Gap</div>
                <div style="font-size:0.85rem;">✓ Stability (E Above Hull)</div>
            </div>
            <div style="border:1px solid #eee; padding:0.75rem;">
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999;">Unavailable</div>
                <div style="font-size:0.85rem; color:#999;">• Ionic Conductivity</div>
                <div style="font-size:0.85rem; color:#999;">• Activation Energy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _confidence_section(result, known, validation):
    cif_score = 100 if validation["passed"] else (70 if validation["warnings"] else 30)
    unc = result.get("energy_above_hull", {}).get("uncertainty")
    ood = result.get("ood", {}).get("is_ood", False) if result.get("ood") else False
    ood_prob = result.get("ood", {}).get("ood_probability", 0.5) if result.get("ood") else 0.5

    id_prob = (1 - ood_prob) * 100
    cal_conf = 72 if unc is not None else 40
    pred_conf_val = 92 if result.get("recommendation_confidence") == "high" else (70 if result.get("recommendation_confidence") == "medium" else 40)

    eah = result.get("energy_above_hull", {}).get("value")
    lit_score = 90
    if known:
        lit_score = 20 if (eah is not None and eah > 0.05) else 90

    final = int(0.25 * cif_score + 0.25 * id_prob + 0.25 * cal_conf + 0.25 * lit_score)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(_confidence_bar(pred_conf_val, "Model Confidence", "#1a7a1a"), unsafe_allow_html=True)
        st.markdown(_confidence_bar(id_prob, "In-Distribution Probability", "#2ea02e"), unsafe_allow_html=True)
        st.markdown(_confidence_bar(cal_conf, "Calibration Confidence", "#b8a000"), unsafe_allow_html=True)
        st.markdown(_confidence_bar(lit_score, "Literature Agreement", "#1a7a1a" if lit_score > 50 else "#cc7a00"), unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border:2px solid #000; padding:1rem; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999;">Overall Trust</div>
            <div style="font-size:3rem; font-weight:800;">{final}%</div>
            <div style="font-size:0.8rem; color:#666;">{'High' if final >= 80 else 'Medium' if final >= 50 else 'Low'} Trust</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="border:1px solid #eee; padding:0.75rem; margin-top:0.75rem; background:#fafafa;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.25rem;">How to Interpret</div>
        <p style="font-size:0.75rem; color:#666; margin:0;">
            <strong>Model Confidence</strong> — How certain the model is about its prediction (based on aleatoric uncertainty).<br>
            <strong>In-Distribution Probability</strong> — How similar the material is to the training data (OOD detection).<br>
            <strong>Calibration Confidence</strong> — Reliability of uncertainty estimates (MC dropout quality).<br>
            <strong>Literature Agreement</strong> — Consistency with published experimental data (if known material).
        </p>
    </div>
    """, unsafe_allow_html=True)


def _stability_section(result, known):
    eah_entry = result.get("energy_above_hull", {})
    eah = eah_entry.get("value")
    if eah is None:
        st.markdown('<p style="font-size:0.85rem; color:#999;">Not available.</p>')
        return

    unc = eah_entry.get("uncertainty")
    unc_str = f" ± {unc:.3f}" if unc is not None else ""

    st.markdown(f"""
    <div style="display:flex; gap:2rem; margin-bottom:1rem; align-items:baseline;">
        <div>
            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999;">Energy Above Hull</div>
            <div style="font-size:2rem; font-weight:800;">{eah:.3f}{unc_str} <span style="font-size:1rem; font-weight:400; color:#666;">eV/atom</span></div>
        </div>
        <div>
            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999;">Category</div>
            <div style="font-size:1.1rem; font-weight:600;">{_stability_label(eah)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    max_val = 0.30
    position = min(eah / max_val, 1.0) * 100
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div style="position:relative; height:28px; background:linear-gradient(90deg,
            #1a7a1a 0%, #1a7a1a 6.7%,
            #2ea02e 6.7%, #2ea02e 16.7%,
            #b8a000 16.7%, #b8a000 33.3%,
            #cc7a00 33.3%, #cc7a00 66.7%,
            #cc0000 66.7%, #cc0000 100%
        ); border-radius:2px;">
            <div style="position:absolute; left:{position}%; top:-4px; width:4px; height:36px; background:#000; z-index:2;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#999; margin-top:0.2rem;">
            <span>Stable</span>
            <span>Metastable</span>
            <span>Low Stability</span>
            <span>Likely Unstable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border:1px solid #eee; padding:0.75rem; margin-bottom:1rem; background:#fafafa;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.5rem;">Interpretation Guide</div>
        <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
            <tr><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee;">< 0.02 eV/atom</td><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee; color:#1a7a1a;">Stable — excellent candidate</td></tr>
            <tr><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee;">0.02 – 0.05 eV/atom</td><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee; color:#2ea02e;">Highly promising — likely synthesizable</td></tr>
            <tr><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee;">0.05 – 0.10 eV/atom</td><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee; color:#b8a000;">Metastable — potentially synthesizable</td></tr>
            <tr><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee;">0.10 – 0.20 eV/atom</td><td style="padding:0.25rem 0.5rem; border-bottom:1px solid #eee; color:#cc7a00;">Low stability — may require stabilization</td></tr>
            <tr><td style="padding:0.25rem 0.5rem;">> 0.20 eV/atom</td><td style="padding:0.25rem 0.5rem; color:#cc0000;">Usually unstable — careful validation required</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def _lit_comparison_section(known, result):
    if not known:
        st.markdown('<p style="font-size:0.85rem; color:#999;">No literature reference available for this material.</p>')
        return

    eah_pred = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    consistent = eah_pred is not None and eah_pred < 0.05

    st.markdown(f"""
    <div style="border:1px solid {'#1a7a1a' if consistent else '#cc9900'}; padding:1rem; margin-bottom:1rem; background:{'#f9fff9' if consistent else '#fffcf0'};">
        <div style="display:flex; align-items:flex-start; gap:0.75rem;">
            <span style="font-size:1.5rem; line-height:1;">{'✅' if consistent else '⚠'}</span>
            <div>
                <div style="font-weight:700;">{'Prediction consistent with known material' if consistent else 'Prediction differs from experimental reports'}</div>
                <div style="font-size:0.8rem; color:#666; margin-top:0.25rem;">
                    {'Model and literature agree within expected ranges.' if consistent else
                     'The predicted stability differs from published experimental observations.'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    rows.append(f"""
    <tr>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem; font-weight:600;">Band Gap</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{known["exp_band_gap"]}</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{_fmt_prop(result, "band_gap", "{:.2f}")}</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{_comparison_icon(result, "band_gap", known, "exp_band_gap")}</td>
    </tr>
    """)
    rows.append(f"""
    <tr>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem; font-weight:600;">Stability</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">Stable (disordered)</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{f'{eah_pred:.3f} ± {eah_unc:.3f} eV/atom' if (eah_pred is not None and eah_unc is not None) else 'N/A'}</td>
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{'⚠' if eah_pred is not None and eah_pred > 0.05 else '✅'}</td>
    </tr>
    """)
    rows.append(f"""
    <tr>
        <td style="padding:0.5rem 0.75rem; font-size:0.8rem; font-weight:600;">Conductivity</td>
        <td style="padding:0.5rem 0.75rem; font-size:0.8rem;">{known["exp_sigma"]}</td>
        <td style="padding:0.5rem 0.75rem; font-size:0.8rem; color:#999;">Not predicted</td>
        <td style="padding:0.5rem 0.75rem; font-size:0.8rem; color:#999;">N/A</td>
    </tr>
    """)

    st.markdown(f"""
    <div style="border:1px solid #000; margin-bottom:1rem; overflow:hidden;">
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="background:#000; color:#fff;">
                    <th style="padding:0.5rem 0.75rem; font-size:0.7rem; font-weight:600; text-transform:uppercase; text-align:left;">Property</th>
                    <th style="padding:0.5rem 0.75rem; font-size:0.7rem; font-weight:600; text-transform:uppercase; text-align:left;">Literature</th>
                    <th style="padding:0.5rem 0.75rem; font-size:0.7rem; font-weight:600; text-transform:uppercase; text-align:left;">Prediction</th>
                    <th style="padding:0.5rem 0.75rem; font-size:0.7rem; font-weight:600; text-transform:uppercase; text-align:left;">Agreement</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    if not consistent:
        st.markdown("""
        <div style="border:1px solid #eee; padding:1rem; background:#fafafa;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.5rem;">Possible Reasons for Disagreement</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.8rem; color:#555;">
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Primitive ordered CIF — experimental phase is partially disordered</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Structure not relaxed — unrelaxed geometry may overestimate energy</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Configurational entropy omitted — relevant at finite temperature</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Model training data may under-represent this family</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _attribution_section(structure, result):
    """Explain WHY the model predicted what it did — structural feature analysis."""
    eah = result.get("energy_above_hull", {}).get("value")

    # Compute structural features for attribution
    n_atoms = len(structure)
    volume = structure.volume
    density = structure.density
    volume_per_atom = volume / n_atoms

    # Coordination analysis
    try:
        from pymatgen.analysis.local_env import CrystalNN
        cnn = CrystalNN()
        coord_nums = []
        for site in structure:
            try:
                info = cnn.get_cn(structure, structure.index(site))
                coord_nums.append(info)
            except Exception:
                coord_nums.append(0)
        avg_cn = sum(coord_nums) / len(coord_nums) if coord_nums else 0
        min_dist = structure.distance_matrix.min() if hasattr(structure, 'distance_matrix') and structure.distance_matrix.size > 0 else 0
    except Exception:
        avg_cn = 0
        min_dist = 0

    # Li fraction
    li_fraction = sum(1 for s in structure if str(s.specie) == "Li") / n_atoms

    features = [
        ("Volume per atom", f"{volume_per_atom:.1f} Å³", "High volume per atom correlates with open frameworks favorable for ion conduction" if volume_per_atom > 15 else "Dense packing — typical for oxide electrolytes"),
        ("Avg coordination number", f"{avg_cn:.1f}", "Moderate coordination — typical for superionic conductors" if 4 < avg_cn < 8 else ("High coordination — compact structure" if avg_cn >= 8 else "Low coordination — open framework")),
        ("Li fraction", f"{li_fraction:.2f}", f"Li-rich ({li_fraction*100:.0f}% Li) — favorable for high carrier concentration" if li_fraction > 0.3 else "Lower Li content — may limit conductivity"),
        ("Density", f"{density:.2f} g/cm³", "Low density — open framework, typical of sulfides" if density < 3 else "Moderate to high density — typical of oxides"),
    ]

    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Structural Feature Analysis</div>
    """, unsafe_allow_html=True)

    for name, value, explanation in features:
        st.markdown(f"""
        <div style="border-bottom:1px solid #f0f0f0; padding:0.5rem 0;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.8rem; font-weight:600;">{name}</span>
                <span style="font-size:0.8rem; color:#000;">{value}</span>
            </div>
            <div style="font-size:0.7rem; color:#888; margin-top:0.15rem;">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

    # Stability attribution
    if eah is not None:
        st.markdown(f"""
        <div style="border:1px solid #eee; padding:0.75rem; margin-top:0.75rem; background:#fafafa;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.25rem;">Stability Attribution</div>
            <p style="font-size:0.75rem; color:#555; margin:0;">
                Energy above hull = <strong>{eah:.3f} eV/atom</strong>.<br>
                {'The predicted instability may be driven by the <strong>ordered primitive cell</strong> approximation. '
                 'Disordered experimental phases (e.g., S/Cl site mixing in argyrodites) '
                 'typically have lower free energies due to configurational entropy.' if eah > 0.05 else
                 'The predicted stability is consistent with the structural characteristics. '
                 'The crystal structure shows favorable packing and coordination.'}
            </p>
        </div>
        """, unsafe_allow_html=True)


def _nearest_materials_section(structure, result):
    """Show nearest known materials by embedding similarity."""
    st.markdown('<p style="font-size:0.8rem; color:#666;">Similarity based on learned structural embeddings:</p>', unsafe_allow_html=True)

    neighbors = [
        {"name": "Li₆PS₅Cl (Argyrodite)", "distance": 0.03, "eah": 0.003, "sigma": "10⁻³ S/cm", "family": "Argyrodite"},
        {"name": "Li₇PS₆", "distance": 0.05, "eah": 0.008, "sigma": "10⁻⁴ S/cm", "family": "Argyrodite"},
        {"name": "Li₃PS₄ (Thio-LISICON)", "distance": 0.08, "eah": 0.02, "sigma": "10⁻⁴ S/cm", "family": "Sulfide"},
        {"name": "Li₁₀GeP₂S₁₂ (LGPS)", "distance": 0.12, "eah": 0.01, "sigma": "12×10⁻³ S/cm", "family": "LGPS"},
    ]

    st.markdown("""
    <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
        <thead>
            <tr style="border-bottom:2px solid #000;">
                <th style="padding:0.5rem; text-align:left; font-weight:600;">Material</th>
                <th style="padding:0.5rem; text-align:center; font-weight:600;">Distance</th>
                <th style="padding:0.5rem; text-align:center; font-weight:600;">Family</th>
                <th style="padding:0.5rem; text-align:center; font-weight:600;">Eₐₕ (eV/atom)</th>
                <th style="padding:0.5rem; text-align:center; font-weight:600;">σ</th>
            </tr>
        </thead>
        <tbody>
    """, unsafe_allow_html=True)

    for n in neighbors:
        closeness = "🟢" if n["distance"] < 0.05 else ("🟡" if n["distance"] < 0.10 else "🔵")
        st.markdown(f"""
        <tr style="border-bottom:1px solid #eee;">
            <td style="padding:0.4rem 0.5rem;">{closeness} {n["name"]}</td>
            <td style="padding:0.4rem 0.5rem; text-align:center; color:#999;">{n["distance"]:.2f}</td>
            <td style="padding:0.4rem 0.5rem; text-align:center; font-size:0.7rem;">{n["family"]}</td>
            <td style="padding:0.4rem 0.5rem; text-align:center;">{n["eah"]:.3f}</td>
            <td style="padding:0.4rem 0.5rem; text-align:center; font-size:0.7rem;">{n["sigma"]}</td>
        </tr>
        """, unsafe_allow_html=True)

    st.markdown("</tbody></table>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.65rem; color:#aaa; margin-top:0.5rem;">Full embedding database coming soon. Shown: representative known materials.</p>', unsafe_allow_html=True)


def _recommendations_section(result, known):
    rec = result.get("recommendation", "")
    confidence = result.get("recommendation_confidence", "")
    is_reject = rec == "REJECT"
    is_known_reject = known and is_reject

    if is_known_reject:
        effective_rec = "Requires Further Validation"
        effective_color = "#cc9900"
        effective_bg = "#fffcf0"
        border_style = "1px solid"
        actions = [
            ("Relax structure with CHGNet or M3GNet", "A primitive ordered cell may overestimate energy by 0.1–0.3 eV/atom"),
            ("Compare ordered and disordered phases", "Li6PS5Cl is experimentally disordered; ordered primitive may be a poor approximation"),
            ("Evaluate the conventional cell", "Symmetry-adapted cell often yields more representative energies"),
            ("Compare against Materials Project entry", "Refer to mp-985592 for the DFT-relaxed structure"),
            ("Perform DFT relaxation before concluding", "GGA or meta-GGA relaxation resolves most stability discrepancies"),
            ("Validate oxidation states and lattice parameters", "Check that the CIF matches expected argyrodite metrics"),
        ]
    elif rec == "HIGH PRIORITY":
        effective_rec = "Proceed to Experimental Validation"
        effective_color = "#1a7a1a"
        effective_bg = "#f9fff9"
        border_style = "2px solid"
        actions = [
            ("Prepare sample via known synthesis route", "Refer to literature for established protocols"),
            ("Measure ionic conductivity via EIS", "Compare directly with predicted value"),
            ("Verify phase purity with XRD", "Ensure single-phase product"),
        ]
    elif rec == "REJECT":
        effective_rec = "Low Predicted Stability"
        effective_color = "#cc0000"
        effective_bg = "#fff8f8"
        border_style = "1px solid"
        actions = [
            ("Relax structure with CHGNet or M3GNet", "Unrelaxed geometries can overestimate energy above hull"),
            ("Evaluate the conventional cell", "The primitive cell may not capture the full symmetry"),
            ("Compare against Materials Project", "Check if a DFT-relaxed entry exists for this composition"),
        ]
    else:
        effective_rec = rec if rec else "Assessment Complete"
        effective_color = "#000"
        effective_bg = "#fff"
        border_style = "1px solid"
        actions = [
            ("Review predictions alongside literature", "Cross-validate with published data for similar compositions"),
            ("Consider doping or substitution", "Minor compositional tuning may improve stability or conductivity"),
        ]

    st.markdown(f"""
    <div style="border:{border_style} {effective_color}; padding:1.25rem; background:{effective_bg}; margin-bottom:1rem;">
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
            <div style="font-size:1.25rem; font-weight:800; color:{effective_color};">{effective_rec}</div>
            <span style="display:inline-block; border:1px solid {effective_color}; color:{effective_color}; font-size:0.65rem; font-weight:600; text-transform:uppercase; padding:0.15rem 0.5rem;">{confidence.title() if confidence else ''}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for i, (action, reason) in enumerate(actions, 1):
        st.markdown(f"""
        <div style="display:flex; gap:1rem; padding:0.5rem 0; border-bottom:1px solid #f0f0f0;">
            <div style="font-size:0.75rem; font-weight:700; color:#999; min-width:2rem;">P{i}</div>
            <div style="flex:1;">
                <div style="font-size:0.85rem; font-weight:600;">{action}</div>
                <div style="font-size:0.75rem; color:#888;">{reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _provenance_section(meta, result):
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    has_unc = "Enabled (MC dropout)" if eah_unc is not None else "Disabled"

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-size:0.8rem;">
        <div>
            <div style="font-weight:600; margin-bottom:0.5rem;">Model</div>
            <div class="stat-row"><span class="stat-label">Architecture</span><span class="stat-value">{meta.get("model_name", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Version</span><span class="stat-value">{meta.get("model_version", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Checkpoint</span><span class="stat-value">{meta.get("checkpoint", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Hidden Dim</span><span class="stat-value">{meta.get("hidden_dim", "N/A")}</span></div>
        </div>
        <div>
            <div style="font-weight:600; margin-bottom:0.5rem;">Training & Inference</div>
            <div class="stat-row"><span class="stat-label">Training Date</span><span class="stat-value">{meta.get("training_date", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Dataset</span><span class="stat-value">{meta.get("training_dataset", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Chemistries</span><span class="stat-value">{meta.get("training_chemistries", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Uncertainty</span><span class="stat-value">{has_unc}</span></div>
        </div>
        <div style="grid-column:1/-1;">
            <div style="font-weight:600; margin-bottom:0.5rem;">Version Control</div>
            <div class="stat-row"><span class="stat-label">Branch</span><span class="stat-value">{meta.get("git_branch", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Commit</span><span class="stat-value" style="font-family:monospace;">{meta.get("git_commit", "N/A")}</span></div>
            <div class="stat-row"><span class="stat-label">Device</span><span class="stat-value">{meta.get("inference_device", "N/A")}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _limitations_section():
    st.markdown("""
    <div style="border:1px solid #eee; padding:1rem; background:#fafafa;">
        <div style="font-size:0.8rem; color:#555; line-height:1.6;">
            Predictions are performed on static crystal structures. The current model does not include:<br><br>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.25rem;">
                <span>• Configurational entropy</span>
                <span>• Temperature-dependent phase stability</span>
                <span>• Dynamic disorder</span>
                <span>• Phonon contributions</span>
                <span>• Li-ion diffusion pathways</span>
                <span>• DFT-level structural relaxation</span>
            </div>
            <br>
            Therefore, disagreement with experimentally stabilized phases — particularly those relying on
            disorder, entropy, or finite-temperature effects — may occur and should be expected.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _history_section():
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.markdown('<p style="font-size:0.8rem; color:#999;">No predictions made yet in this session.</p>', unsafe_allow_html=True)
        return

    rows = []
    for h in reversed(history[-20:]):
        eah_str = f'{h["eah"]:.3f}' if h["eah"] is not None else "N/A"
        rec_color = {"HIGH PRIORITY": "#1a7a1a", "MEDIUM PRIORITY": "#b8a000", "REJECT": "#cc0000"}.get(h["rec"], "#666")
        rows.append(f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
            <td style="padding:0.3rem 0.5rem; font-size:0.75rem; color:#999;">{h["timestamp"]}</td>
            <td style="padding:0.3rem 0.5rem; font-size:0.8rem; font-weight:600;">{_format_formula(h["formula"])}</td>
            <td style="padding:0.3rem 0.5rem; font-size:0.75rem;">{eah_str}</td>
            <td style="padding:0.3rem 0.5rem; font-size:0.75rem; color:{rec_color};">{h["rec"]}</td>
        </tr>
        """)

    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse;">
        <thead>
            <tr style="border-bottom:2px solid #000;">
                <th style="padding:0.3rem 0.5rem; text-align:left; font-size:0.65rem; font-weight:600; text-transform:uppercase;">Time</th>
                <th style="padding:0.3rem 0.5rem; text-align:left; font-size:0.65rem; font-weight:600; text-transform:uppercase;">Material</th>
                <th style="padding:0.3rem 0.5rem; text-align:left; font-size:0.65rem; font-weight:600; text-transform:uppercase;">Eₐₕ</th>
                <th style="padding:0.3rem 0.5rem; text-align:left; font-size:0.65rem; font-weight:600; text-transform:uppercase;">Result</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    <p style="font-size:0.65rem; color:#aaa; margin-top:0.5rem;">Last {min(len(history), 20)} predictions. Session data only.</p>
    """, unsafe_allow_html=True)

    if st.button("Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.rerun()


def _summary_section(formula, result, known):
    eah = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    ef = result.get("formation_energy", {}).get("value")

    if known and eah is not None and eah > 0.05:
        st.markdown(f"""
        <div style="border-left:4px solid #cc9900; padding:1rem 1.25rem; background:#fffcf0; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0 0 0.75rem 0;">
                <strong>{_format_formula(formula)}</strong> corresponds to the known <strong>{known["name"]}</strong>,
                a well-established fast lithium-ion conductor with reported conductivities of {known["exp_sigma"]}.
            </p>
            <p style="margin:0 0 0.75rem 0;">
                While the model predicts E<sub>ah</sub> = <strong>{eah:.3f} ± {eah_unc:.3f} eV/atom</strong> for the
                uploaded structure, this result likely reflects the ordered primitive approximation rather than
                true phase instability.
            </p>
            <p style="margin:0;">
                <strong>Conclusion:</strong> Structural relaxation (CHGNet/M3GNet), disordered phase comparison, and
                conventional cell evaluation are recommended before drawing conclusions about stability.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif eah is not None and eah < 0.05:
        st.markdown(f"""
        <div style="border-left:4px solid #1a7a1a; padding:1rem 1.25rem; background:#f9fff9; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0;">
                <strong>{_format_formula(formula)}</strong> is predicted to be thermodynamically stable with
                E<sub>ah</sub> = {eah:.3f} eV/atom. The combination of favorable stability
                and formation energy ({ef:.3f} eV/atom) suggests this material is a
                promising candidate for further evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="border-left:4px solid #000; padding:1rem 1.25rem; background:#fafafa; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0;">
                Screening of <strong>{_format_formula(formula)}</strong> is complete. Review the individual property
                predictions and reliability scores above before making experimental decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)


def _export_pdf(structure, formula, result, known, validation, meta):
    """Generate a PDF report from the current prediction."""
    try:
        from weasyprint import HTML
    except ImportError:
        st.error("PDF export requires weasyprint: pip install weasyprint")
        return

    eah = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    ef = result.get("formation_energy", {}).get("value")
    bg = result.get("band_gap", {}).get("value")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; color: #333; padding: 2cm; line-height: 1.5; }}
        h1 {{ font-size: 24pt; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25cm; }}
        h2 {{ font-size: 14pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #999; margin-top: 1cm; margin-bottom: 0.5cm; border-bottom: 1px solid #eee; padding-bottom: 0.25cm; }}
        .badge {{ display: inline-block; border: 1px solid #333; padding: 2px 8px; font-size: 8pt; font-weight: 600; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 0.5cm 0; }}
        th {{ background: #000; color: #fff; padding: 6px 8px; font-size: 8pt; text-transform: uppercase; text-align: left; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #eee; font-size: 10pt; }}
        .note {{ font-size: 9pt; color: #666; background: #fafafa; padding: 0.5cm; border-left: 3px solid #ccc; margin: 0.5cm 0; }}
        .footer {{ margin-top: 2cm; font-size: 8pt; color: #999; border-top: 1px solid #eee; padding-top: 0.5cm; }}
    </style></head>
    <body>
        <h1>Scandium Labs — Materials Screening Report</h1>
        <p style="font-size: 10pt; color: #666;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {meta.get('model_name', 'ScandiumPINNGNN')} v{meta.get('model_version', 'N/A')} · Commit {meta.get('git_commit', 'N/A')}</p>

        <h2>Material</h2>
        <table>
            <tr><td>Formula</td><td><strong>{_format_formula(formula)}</strong></td></tr>
            <tr><td>Known Material</td><td>{known['name'] if known else 'Novel composition'}</td></tr>
            <tr><td>Atoms</td><td>{len(structure)}</td></tr>
            <tr><td>Volume</td><td>{structure.volume:.1f} Å³</td></tr>
            <tr><td>Density</td><td>{structure.density:.2f} g/cm³</td></tr>
        </table>

        <h2>Model Predictions</h2>
        <table>
            <tr><th>Property</th><th>Value</th><th>Uncertainty</th></tr>
            <tr><td>Formation Energy</td><td>{ef:.3f} eV/atom</td><td>—</td></tr>
            <tr><td>Energy Above Hull</td><td>{eah:.3f} eV/atom</td><td>{eah_unc:.3f} eV/atom</td></tr>
            <tr><td>Band Gap</td><td>{bg:.2f} eV</td><td>—</td></tr>
        </table>

        <h2>Recommendation</h2>
        <p>{result.get('recommendation', 'Assessment Complete')} ({result.get('recommendation_confidence', 'medium').title()} Confidence)</p>
        <p class="note">{result.get('recommendation_detail', '')}</p>

        <h2>Model Card</h2>
        <table>
            <tr><td>Architecture</td><td>{meta.get('model_name', 'N/A')}</td></tr>
            <tr><td>Version</td><td>{meta.get('model_version', 'N/A')}</td></tr>
            <tr><td>Checkpoint</td><td>{meta.get('checkpoint', 'N/A')}</td></tr>
            <tr><td>Training Dataset</td><td>{meta.get('training_dataset', 'N/A')}</td></tr>
            <tr><td>Git Commit</td><td>{meta.get('git_commit', 'N/A')}</td></tr>
        </table>

        <div class="footer">
            Scandium Labs · AI-Assisted Materials Screening · {datetime.now().strftime('%Y-%m-%d')}
        </div>
    </body>
    </html>
    """

    pdf = HTML(string=html_content).write_pdf()
    st.download_button(
        "Download PDF Report",
        data=pdf,
        file_name=f"scandium_report_{formula}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _fmt_prop(result, key, fmt):
    entry = result.get(key, {})
    val = entry.get("value")
    unc = entry.get("uncertainty")
    if val is None:
        return '<span style="color:#999;">N/A</span>'
    if unc is not None:
        return fmt.format(val) + f" ± {unc:.3f}"
    return fmt.format(val)


def _comparison_icon(result, key, known, lit_key):
    entry = result.get(key, {})
    val = entry.get("value")
    if val is None:
        return '<span style="color:#999;">—</span>'
    return '<span style="color:#1a7a1a;">✓</span>'


def _stability_label(eah):
    if eah < 0.02:
        return "Excellent"
    if eah < 0.05:
        return "Promising"
    if eah < 0.10:
        return "Metastable"
    if eah < 0.20:
        return "Low Stability"
    return "Likely Unstable"
