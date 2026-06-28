import streamlit as st

st.set_page_config(
    page_title="Scandium Labs",
    page_icon="▤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

.stApp {
    background: #ffffff;
    color: #000000;
}

.stApp > header { display: none; }

.main > .block-container {
    max-width: 1200px;
    padding: 3rem 4rem;
}

h1 {
    font-size: 4.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    line-height: 0.95 !important;
    color: #000000 !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #000000 !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    color: #000000 !important;
}

p, li, .stMarkdown {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #333333 !important;
}

.stButton > button {
    background: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: none !important;
}

.stButton > button:hover {
    background: #cc0000 !important;
    color: #ffffff !important;
}

.stButton > button:disabled {
    background: #cccccc !important;
    color: #888888 !important;
}

div.stFileUploader {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    background: #ffffff !important;
    padding: 0 !important;
}

div.stFileUploader > section {
    border: none !important;
    padding: 2rem !important;
    background: #ffffff !important;
}

div.stFileUploader:hover {
    background: #fafafa !important;
}

.stTextInput > div > div,
.stTextArea > div > div,
.stNumberInput > div > div {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: #ffffff !important;
}

.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: #cc0000 !important;
}

.stSelectbox > div > div {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
}

div[data-testid="stMetric"] {
    border: none !important;
    background: none !important;
    padding: 0 !important;
}

div[data-testid="stMetric"] > div {
    border: none !important;
}

div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #666666 !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #000000 !important;
}

.stDataFrame {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
}

.stDataFrame table {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}

.stDataFrame thead tr th {
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-size: 0.75rem !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid #000000 !important;
}

.stDataFrame tbody tr td {
    padding: 0.6rem 1rem !important;
    border-bottom: 1px solid #eeeeee !important;
    color: #000000 !important;
}

.stDataFrame tbody tr:hover {
    background: #f5f5f5 !important;
}

.stProgress > div > div {
    background: #000000 !important;
    border-radius: 0 !important;
}

.stProgress > div {
    background: #eeeeee !important;
    border-radius: 0 !important;
}

.stAlert {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    background: #ffffff !important;
    padding: 1.25rem !important;
}

.stAlert p { margin: 0 !important; }

div[data-testid="stAlertSuccess"] {
    border-left: 4px solid #000000 !important;
}

div[data-testid="stAlertError"] {
    border-left: 4px solid #cc0000 !important;
}

div[data-testid="stAlertWarning"] {
    border-left: 4px solid #000000 !important;
}

div[data-testid="stAlertInfo"] {
    border-left: 4px solid #000000 !important;
}

.stRadio > div {
    gap: 0 !important;
}

.stRadio label {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    padding: 0.4rem 1.2rem !important;
    margin-right: 0 !important;
    background: #ffffff !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}

.stRadio label[data-checked="true"] {
    background: #000000 !important;
    color: #ffffff !important;
}

.stCheckbox label {
    font-size: 0.85rem !important;
}

.sidebar-section {
    margin-bottom: 3rem;
}

.sidebar-section .label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #999999;
    margin-bottom: 0.5rem;
}

.divider {
    width: 100%;
    height: 1px;
    background: #000000;
    margin: 2.5rem 0;
}

.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 2rem;
}

.grid-4 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 2rem;
}

.asymmetric {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 3rem;
}

.card {
    border: 1px solid #000000;
    padding: 1.5rem;
}

.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #999999;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #000000;
}

.card-value.red {
    color: #cc0000;
}

.red-accent {
    color: #cc0000;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid #eeeeee;
    padding: 0.75rem 0;
}

.stat-row:last-child {
    border-bottom: none;
}

.stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #666666;
}

.stat-value {
    font-size: 1rem;
    font-weight: 600;
    color: #000000;
}

.bar-container {
    width: 100%;
    height: 8px;
    background: #eeeeee;
    margin: 0.5rem 0;
}

.bar-fill {
    height: 8px;
    background: #000000;
}

.bar-fill.red {
    background: #cc0000;
}

.pill {
    display: inline-block;
    border: 1px solid #000000;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.pill.red {
    border-color: #cc0000;
    color: #cc0000;
}

.pill.gray {
    border-color: #cccccc;
    color: #999999;
}

.sidebar-nav {
    list-style: none;
    padding: 0;
    margin: 0;
}

.sidebar-nav li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #eeeeee;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
}

.sidebar-nav li.active {
    font-weight: 700;
    color: #cc0000;
}
</style>
""", unsafe_allow_html=True)

PAGES = {
    "Dashboard": "01",
    "Screen": "02",
    "Batch": "03",
    "Results": "04",
}

def main():
    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap:1.5rem; margin-bottom:3rem;">
        <span style="font-size:2rem; font-weight:900; letter-spacing:-0.03em; color:#000;">Scandium</span>
        <span style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#999;">Labs</span>
        <span style="flex:1;"></span>
        <span style="font-size:0.7rem; font-weight:500; text-transform:uppercase; letter-spacing:0.05em; color:#999;">Solid Electrolyte Discovery</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 4])
    page_map = list(PAGES.keys())
    for i, (name, num) in enumerate(PAGES.items()):
        with cols[i]:
            active = st.session_state.get("page", "Dashboard") == name
            border = "2px solid #cc0000" if active else "1px solid #ddd"
            bg = "#ffffff" if not active else "#ffffff"
            if st.button(f"{num}. {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state["page"] = name
                st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    page = st.session_state.get("page", "Dashboard")

    if page == "Dashboard":
        import pages.dashboard
        pages.dashboard.show()
    elif page == "Screen":
        import pages.screen
        pages.screen.show()
    elif page == "Batch":
        import pages.batch
        pages.batch.show()
    elif page == "Results":
        import pages.results
        pages.results.show()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#999; text-transform:uppercase; letter-spacing:0.05em;">
        <span>Scandium Labs — v1.0</span>
        <span>PINN-GNN · Solid Electrolyte Screening</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
