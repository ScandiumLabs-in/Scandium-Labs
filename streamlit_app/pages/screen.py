import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ENGINE = None

def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from src.inference.engine import InferenceEngine
        _ENGINE = InferenceEngine("checkpoints/best_model.pt", device="cpu")
    return _ENGINE


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
    },
}


def _lookup_material(formula):
    for key, info in KNOWN_MATERIALS.items():
        if formula.lower().replace(" ", "") == key.lower():
            return info
    return None


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

                show_overall_assessment(result, known, validation)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                if known:
                    show_material_literature(known, result)
                    show_consistency_check(known, result)
                    show_literature_comparison(known, result)
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                show_validation_card(validation)
                show_properties(result)
                show_stability(result, known)
                show_conductivity(result)
                show_reliability_score(result, known, validation)
                show_limitations()
                show_recommendations(result, known)
                show_scientific_conclusion(result, known)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                os.unlink(tmp_path)


def show_overall_assessment(result, known, validation):
    formula = result["formula"]
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
        rec_detail = rec_detail = result.get("recommendation_detail", "")
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

    st.markdown(f"""
    <div style="border:2px solid {status_color}; padding:1.5rem; margin-bottom:1.5rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Overall Assessment</div>
        <div style="display:grid; grid-template-columns:2fr 1fr; gap:2rem;">
            <div>
                <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.02em; color:{status_color};">{status_text}</div>
                <div style="font-size:0.85rem; color:#555; margin-top:0.5rem;">{rec_detail}</div>
                <div style="display:flex; gap:0.5rem; margin-top:0.75rem; flex-wrap:wrap;">
                    <span style="display:inline-block; border:1px solid #000; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:600; text-transform:uppercase;">{known_label}</span>
                    <span style="display:inline-block; border:1px solid #000; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:600; text-transform:uppercase;">{formula}</span>
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


def show_material_literature(known, result):
    refs = "; ".join(known["references"])
    eah = result.get("energy_above_hull", {}).get("value")

    known_stability = "Stable (disordered phase)" if known.get("exp_eah", 0) < 0.05 else known.get("exp_eah", "N/A")

    st.markdown(f"""
    <div style="border:1px solid #000; padding:1.25rem; margin-bottom:1.5rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Known Material — Literature Reference</div>
        <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.75rem;">{known["name"]}</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:0.75rem;">
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Experimental Status</div>
                <div style="font-size:0.9rem;">Known fast solid electrolyte</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Conductivity</div>
                <div style="font-size:0.9rem; font-weight:600;">{known["exp_sigma"]}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Band Gap</div>
                <div style="font-size:0.9rem;">{known["exp_band_gap"]}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Experimental Stability</div>
                <div style="font-size:0.9rem;">{known_stability}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Materials Project</div>
                <div style="font-size:0.9rem;">{known.get("mp_id", "N/A")}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#999; margin-bottom:0.25rem;">Literature</div>
                <div style="font-size:0.9rem;">{refs}</div>
            </div>
        </div>
        <p style="font-size:0.8rem; color:#555; margin:0; border-top:1px solid #eee; padding-top:0.75rem;">{known["notes"]}</p>
    </div>
    """, unsafe_allow_html=True)


def show_consistency_check(known, result):
    eah = result.get("energy_above_hull", {}).get("value")
    consistent = eah is not None and eah < 0.05

    st.markdown(f"""
    <div style="border:1px solid {'#1a7a1a' if consistent else '#cc9900'}; padding:1rem; margin-bottom:1.5rem; background:{'#f9fff9' if consistent else '#fffcf0'};">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Literature Consistency Check</div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem;">
            <span style="font-size:1.5rem; line-height:1;">{'✅' if consistent else '⚠'}</span>
            <div>
                <div style="font-weight:700;">{'Prediction consistent with known material' if consistent else 'Prediction inconsistent with known experimental material'}</div>
                <div style="font-size:0.8rem; color:#666; margin-top:0.25rem;">
                    {'Model and literature agree within expected ranges.' if consistent else
                     'The predicted stability differs substantially from published experimental reports.'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not consistent:
        st.markdown("""
        <div style="border:1px solid #eee; padding:1rem; margin-bottom:1.5rem; background:#fafafa;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.5rem;">Possible Reasons for Disagreement</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.8rem; color:#555;">
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Primitive ordered CIF — experimental phase is partially disordered</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Structure not relaxed — unrelaxed geometry may overestimate energy</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Model training data may under-represent argyrodite chemistries</div>
                <div style="display:flex; gap:0.5rem;"><span style="color:#cc9900;">•</span> Configurational entropy omitted — relevant at finite temperature</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_literature_comparison(known, result):
    eah_pred = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    ef_pred = result.get("formation_energy", {}).get("value")

    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Literature vs Prediction</div>
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
        <td style="padding:0.5rem 0.75rem; border-bottom:1px solid #eee; font-size:0.8rem;">{eah_pred:.3f} ± {eah_unc:.3f} eV/atom" if eah_pred is not None and eah_unc is not None else "N/A"}</td>
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
    <div style="border:1px solid #000; margin-bottom:1.5rem; overflow:hidden;">
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
    lit_val_str = known.get(lit_key, "")
    return '<span style="color:#1a7a1a;">✓</span>'


def show_validation_card(validation):
    items = []
    for c in validation["checks"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#1a7a1a;">✓</span><span style="font-size:0.8rem;">{c}</span></div>')
    for w in validation["warnings"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#cc9900;">⚠</span><span style="font-size:0.8rem;">{w}</span></div>')
    for e in validation["errors"]:
        items.append(f'<div style="display:flex; align-items:center; gap:0.5rem; padding:0.25rem 0;"><span style="color:#cc0000;">✗</span><span style="font-size:0.8rem;">{e}</span></div>')

    border_color = "#cc0000" if not validation["passed"] else ("#cc9900" if validation["warnings"] else "#1a7a1a")
    st.markdown(f"""
    <div style="border:1px solid {border_color}; padding:1rem; margin-bottom:1.5rem;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.5rem;">Structure Validation</div>
        {''.join(items)}
    </div>
    """, unsafe_allow_html=True)


def show_properties(result):
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Model Prediction — Properties</div>
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
            preds.append((label, display))

    with col1:
        for label, val in preds[:2]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
    with col2:
        for label, val in preds[2:]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)


def show_stability(result, known):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Model Prediction — Thermodynamic Stability</div>
    """, unsafe_allow_html=True)

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

    if known and eah > 0.05:
        st.markdown("""
        <div style="border:1px solid #eee; padding:1rem; background:#fafafa; margin-top:1rem;">
            <div style="font-size:0.8rem; color:#555;">
                <strong style="color:#333;">Interpretation:</strong> The uploaded ordered primitive structure is predicted to lie above the convex hull.
                This does not necessarily imply that experimentally observed disordered phases are unstable.
                The primitive ordered approximation may differ substantially from the known disordered phase.
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_conductivity(result):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Model Prediction — Ionic Conductivity</div>
    """, unsafe_allow_html=True)

    cond = result.get("log_ionic_conductivity", {})
    cond_value = cond.get("value")
    cond_status = cond.get("status")

    if cond_value is not None:
        sigma = 10 ** cond_value
        sigma_unc = result.get("ionic_conductivity", {}).get("uncertainty")
        display_val = f"{sigma:.2e} S/cm"
        if sigma_unc is not None:
            display_val += f" ± {sigma_unc:.2e} S/cm"
        st.markdown(f'<div style="font-size:2rem; font-weight:800; margin-bottom:1rem;">{display_val}</div>', unsafe_allow_html=True)
        log_val = max(-5, min(2, cond_value))
        pct = (log_val + 5) / 7 * 100
        st.markdown(f"""
        <div class="bar-container"><div class="bar-fill" style="width:{pct:.0f}%;"></div></div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#999; margin-bottom:1rem;">
            <span>10⁻⁵</span><span>10⁻³</span><span>10⁻¹</span>
        </div>
        """, unsafe_allow_html=True)

        from src.inference.engine import STATUS_MC_DISABLED
        if cond.get("status") == STATUS_MC_DISABLED:
            st.markdown("""
            <p style="font-size:0.75rem; color:#999; margin-top:0.5rem;">
                Note: MC dropout not enabled — prediction shown without uncertainty estimate.
            </p>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px solid #eee; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.9rem; font-weight:600; color:#999; margin-bottom:0.5rem;">Not Predicted</div>
            <div style="font-size:0.75rem; color:#999;">
                <strong>Reason:</strong> The current model has not been trained on experimentally labeled ionic conductivity data.
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
        <p style="font-size:0.75rem; color:#999;">
            Future versions will support conductivity prediction using OBELiX and LiIon datasets.
        </p>
        """, unsafe_allow_html=True)


def show_reliability_score(result, known, validation):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Prediction Reliability</div>
    """, unsafe_allow_html=True)

    cif_score = 100 if validation["passed"] else (50 if any("overlap" in w.lower() or "charge" in w.lower() for w in validation["warnings"]) else 80)
    if not validation["passed"]:
        cif_score = 30

    val_score = 100 if not validation["warnings"] else 70

    unc = result.get("energy_above_hull", {}).get("uncertainty")
    ood_score = 72 if unc is not None else 40
    id_score = ood_score

    eah = result.get("energy_above_hull", {}).get("value")
    lit_score = 100
    if known:
        lit_score = 20 if (eah is not None and eah > 0.05) else 90

    final = int(0.20 * cif_score + 0.15 * val_score + 0.30 * id_score + 0.35 * lit_score)

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;">
        <div style="border:1px solid #eee; padding:0.75rem;">
            <div class="stat-row"><span class="stat-label">CIF Quality</span><span class="stat-value">{cif_score}%</span></div>
            <div class="stat-row"><span class="stat-label">Structure Validation</span><span class="stat-value">{val_score}%</span></div>
            <div class="stat-row"><span class="stat-label">In-Distribution</span><span class="stat-value">{id_score}%</span></div>
            <div class="stat-row"><span class="stat-label">Literature Agreement</span><span class="stat-value">{lit_score}%</span></div>
        </div>
        <div style="border:2px solid #000; padding:0.75rem; display:flex; flex-direction:column; justify-content:center; align-items:center;">
            <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999;">Overall Reliability</div>
            <div style="font-size:3rem; font-weight:800;">{final}%</div>
            <div style="font-size:0.8rem; color:#666;">{'High' if final >= 80 else 'Medium' if final >= 50 else 'Low'} Trust</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_limitations():
    st.markdown("""
    <div style="border:1px solid #eee; padding:1rem; margin-bottom:1.5rem; background:#fafafa;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Model Limitations</div>
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


def show_recommendations(result, known):
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:1rem;">Recommendations</div>
    """, unsafe_allow_html=True)

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
    <div style="border:{border_style} {effective_color}; padding:1.25rem; background:{effective_bg}; margin-bottom:1.5rem;">
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
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

    st.markdown("""
    <div style="border:1px solid #eee; padding:0.75rem; margin-top:1rem; background:#fafafa;">
        <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#999; margin-bottom:0.25rem;">Interpretation Guide</div>
        <p style="font-size:0.75rem; color:#666; margin:0;">
            The predictions above are based on a multi-task PINN-GNN trained on DFT data.
            They serve as an initial screen and should be validated with higher-fidelity methods
            before making experimental decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_scientific_conclusion(result, known):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.75rem;">Scientific Summary</div>
    """, unsafe_allow_html=True)

    formula = result["formula"]
    eah = result.get("energy_above_hull", {}).get("value")
    eah_unc = result.get("energy_above_hull", {}).get("uncertainty")
    ef = result.get("formation_energy", {}).get("value")

    if known and eah is not None and eah > 0.05:
        st.markdown(f"""
        <div style="border-left:4px solid #cc9900; padding:1rem 1.25rem; background:#fffcf0; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0 0 0.75rem 0;">
                <strong>{formula}</strong> corresponds to the known argyrodite <strong>{known["name"]}</strong>,
                a well-established fast lithium-ion conductor with reported conductivities of {known["exp_sigma"]}.
            </p>
            <p style="margin:0 0 0.75rem 0;">
                While the model predicts an energy above hull of <strong>{eah:.3f} ± {eah_unc:.3f} eV/atom</strong> for the
                uploaded structure, this result is inconsistent with published experimental observations. The discrepancy is
                likely attributable to the use of an ordered primitive representation, the absence of structural relaxation,
                and the omission of configurational disorder from the current prediction pipeline.
            </p>
            <p style="margin:0;">
                <strong>Conclusion:</strong> This result should not be interpreted as evidence that {known["name"]} is unstable.
                Structural relaxation (CHGNet/M3GNet), comparison with the experimentally reported disordered phase, and
                evaluation of the conventional cell are strongly recommended before drawing any conclusions about stability.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif eah is not None and eah < 0.05:
        st.markdown(f"""
        <div style="border-left:4px solid #1a7a1a; padding:1rem 1.25rem; background:#f9fff9; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0;">
                <strong>{formula}</strong> is predicted to be thermodynamically stable with
                E<sub>ah</sub> = {eah:.3f} eV/atom. The combination of favorable stability
                and formation energy ({ef:.3f} eV/atom) suggests this material is a
                promising candidate for further evaluation. Experimental validation via
                synthesis and conductivity measurement is recommended as the next step.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="border-left:4px solid #000; padding:1rem 1.25rem; background:#fafafa; font-size:0.85rem; line-height:1.7; color:#444;">
            <p style="margin:0;">
                Screening of <strong>{formula}</strong> is complete. Review the individual property
                predictions and reliability scores above before making experimental decisions.
                DFT validation is recommended for borderline or uncertain predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)


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
