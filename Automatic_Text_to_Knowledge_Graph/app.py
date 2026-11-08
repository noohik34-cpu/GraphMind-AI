import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd
import json
import fitz
import tempfile
import os

from pyvis.network import Network
from graph_generator import (
    extract_knowledge,
    graph_statistics,
    get_relationships,
    answer_question
)

# PAGE CONFIG
st.set_page_config(
    page_title="GraphMind AI",
    page_icon="🧠",
    layout="wide"
)

# PROFESSIONAL THEME + TEXT FIX
st.markdown("""
<style>
.stApp {background:#0b1020;color:#fff}
[data-testid="stSidebar"] {background:#111a30}
h1,h2,h3,h4 {color:#f8fafc!important}
p,label {color:#e2e8f0}
.stTextArea textarea,.stTextInput input {
    background:white!important;
    color:#111827!important;
    -webkit-text-fill-color:#111827!important;
    caret-color:#111827!important;
}
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color:#64748b!important;
    -webkit-text-fill-color:#64748b!important;
}
[data-testid="stFileUploader"] {
    background:white;padding:10px;border-radius:10px
}
[data-testid="stFileUploader"] * {color:#111827!important}
.stButton button {
    background:linear-gradient(90deg,#7c3aed,#4f46e5);
    color:white!important;border-radius:9px;
    border:0;font-weight:bold
}
.stButton button p {color:white!important}
[data-testid="stMetric"] {
    background:#182644;padding:15px;
    border:1px solid #39476b;border-radius:12px
}
[data-testid="stMetricValue"] {color:#a78bfa}
[data-testid="stMetricLabel"] {color:white}
[data-baseweb="select"]>div {
    background:white;color:#111827
}
[data-baseweb="select"] span {color:#111827}
.hero {
    background:linear-gradient(110deg,#172554,#4c1d95);
    padding:25px;border-radius:15px;
    border:1px solid #5145a8;margin-bottom:20px
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if "graph" not in st.session_state:
    st.session_state.graph = None
if "source" not in st.session_state:
    st.session_state.source = ""

# SIDEBAR
with st.sidebar:
    st.markdown("# 🧠 GraphMind AI")
    st.caption("KNOWLEDGE INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio("NAVIGATION", [
        "Dashboard",
        "Knowledge Graph",
        "Entity Explorer",
        "Relationship Explorer",
        "AI Assistant",
        "About Project"
    ])

    st.divider()
    st.markdown("""
    🟢 NLP Processing

    🟢 Entity Recognition

    🟢 Relationship Extraction

    🟢 Duplicate Normalization

    🟢 Confidence Scoring

    🟢 Evidence Tracking
    """)

# HEADER
st.markdown("""
<div class="hero">
<h1>🧠 GraphMind AI</h1>
<h3>Automatic Text-to-Knowledge Graph Generator</h3>
<p>Convert unstructured text into intelligent,
connected and interactive knowledge.</p>
</div>
""", unsafe_allow_html=True)

# DOCUMENT INPUT
with st.expander("📂 DOCUMENT PROCESSING CENTER", expanded=True):

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    sample = """Elon Musk is the CEO of Tesla.
Tesla is an automobile company.
Elon Musk founded SpaceX.
SpaceX develops rocket technology.
NASA collaborates with SpaceX.
NASA is headquartered in Washington.
OpenAI develops artificial intelligence.
Sam Altman leads OpenAI."""

    text = st.text_area(
        "Enter Your Text",
        value=st.session_state.source or sample,
        height=150
    )

    if uploaded:
        try:
            pdf = fitz.open(
                stream=uploaded.getvalue(),
                filetype="pdf"
            )
            pdf_text = "\n".join(p.get_text() for p in pdf)
            pdf.close()
            if pdf_text.strip():
                text = pdf_text
                st.success("PDF text extracted!")
            else:
                st.warning("No readable PDF text found.")
        except Exception as e:
            st.error(str(e))

    if st.button("⚡ GENERATE KNOWLEDGE GRAPH",
                 type="primary", use_container_width=True):
        if text.strip():
            try:
                with st.spinner("Processing text..."):
                    graph, entities = extract_knowledge(text)
                    st.session_state.graph = graph
                    st.session_state.source = text
                st.success("Knowledge graph generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Processing error: {e}")
        else:
            st.warning("Please enter text.")

# GRAPH CHECK
graph = st.session_state.graph

if graph is None:
    st.info("Enter text or upload PDF to generate your graph.")
    st.stop()

# GRAPH DATA
stats = graph_statistics(graph)

entities = pd.DataFrame([
    {
        "ID": n,
        "Entity": d.get("label", n),
        "Category": d.get("entity_type", "UNKNOWN"),
        "Aliases": d.get("mentions", "")
    }
    for n, d in graph.nodes(data=True)
])

relations = pd.DataFrame(
    get_relationships(graph),
    columns=["Source", "Relationship", "Target",
             "Confidence", "Evidence"]
)

# DASHBOARD
if page == "Dashboard":
    st.subheader("📊 Knowledge Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Entities", stats["entities"])
    c2.metric("Relationships", stats["relationships"])
    c3.metric("Graph Density", stats["density"])
    c4.metric("Components", stats["components"])

    a,b = st.columns(2)
    with a:
        st.markdown("### 🏷️ Entity Distribution")
        if not entities.empty:
            st.bar_chart(entities["Category"].value_counts())

    with b:
        st.markdown("### 🔗 Relationship Distribution")
        if not relations.empty:
            st.bar_chart(relations["Relationship"].value_counts())

    st.markdown("### Extracted Relationships")
    st.dataframe(relations, use_container_width=True,
                 hide_index=True)

# INTERACTIVE GRAPH
elif page == "Knowledge Graph":
    st.subheader("🌐 Interactive Knowledge Graph")

    categories = sorted(entities["Category"].unique())
    selected = st.multiselect(
        "Filter Categories", categories, default=categories
    )
    confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0)

    nodes = entities[entities.Category.isin(selected)]["ID"].tolist()
    g = nx.DiGraph()

    for n in nodes:
        g.add_node(n, **graph.nodes[n])

    for u,v,d in graph.edges(data=True):
        if u in nodes and v in nodes and d["confidence"] >= confidence:
            g.add_edge(u,v,**d)

    net = Network(
        height="580px", width="100%",
        directed=True, bgcolor="#0b1020",
        font_color="white", cdn_resources="in_line"
    )

    colors = {
        "PERSON":"#a78bfa",
        "ORG":"#38bdf8",
        "GPE":"#34d399",
        "LOC":"#fbbf24"
    }

    for n,d in g.nodes(data=True):
        net.add_node(
            n,
            label=d["label"],
            title=d.get("entity_type",""),
            color=colors.get(d.get("entity_type"),"#818cf8"),
            size=25
        )

    for u,v,d in g.edges(data=True):
        net.add_edge(
            u,v,
            label=d.get("relation","RELATED_TO"),
            title="Evidence: "+d.get("evidence",""),
            arrows="to"
        )

    if g.number_of_nodes():
        net.set_options("""
        {
          "physics":{"enabled":true,"barnesHut":{
            "gravitationalConstant":-3500,"springLength":180
          }},
          "interaction":{"hover":true,"navigationButtons":true}
        }
        """)

        path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False
            ) as f:
                path = f.name

            net.save_graph(path)
            with open(path, encoding="utf-8") as f:
                html = f.read()

            components.html(html, height=600, scrolling=True)

            st.download_button(
                "⬇ Download Interactive HTML",
                html, "knowledge_graph.html",
                "text/html"
            )
        finally:
            if path and os.path.exists(path):
                os.remove(path)
    else:
        st.warning("No matching nodes.")

# ENTITY EXPLORER
elif page == "Entity Explorer":
    st.subheader("🏷️ Entity Explorer")

    search = st.text_input("Search Entity")
    result = entities[
        entities.Entity.str.contains(search, case=False, na=False)
    ]

    st.dataframe(result.drop(columns=["ID"]),
                 use_container_width=True, hide_index=True)

    if not result.empty:
        name = st.selectbox("Select Entity", result.Entity.tolist())
        node = result[result.Entity == name].iloc[0]["ID"]

        st.markdown(f"### Connections of {name}")

        for u,v,d in graph.in_edges(node, data=True):
            st.write(f"{graph.nodes[u]['label']} → {d['relation']} → {name}")

        for u,v,d in graph.out_edges(node, data=True):
            st.write(f"{name} → {d['relation']} → {graph.nodes[v]['label']}")

# RELATIONSHIP EXPLORER
elif page == "Relationship Explorer":
    st.subheader("🔗 Relationship Explorer")

    search = st.text_input("Search Relationship")

    result = relations[
        relations.astype(str).apply(
            lambda r: r.str.contains(
                search, case=False, na=False
            ).any(), axis=1
        )
    ]

    st.dataframe(result, use_container_width=True, hide_index=True)

    if not relations.empty:
        st.markdown("### Confidence Analysis")
        st.bar_chart(relations["Confidence"])
        st.metric("Average Heuristic Confidence",
                  f"{relations['Confidence'].mean():.0%}")

# AI ASSISTANT
elif page == "AI Assistant":
    st.subheader("🤖 Knowledge Graph Assistant")

    question = st.text_input(
        "Ask a question",
        placeholder="Which company is connected to Elon Musk?"
    )

    if st.button("🔍 SEARCH"):
        st.info(
            answer_question(graph, question)
            if question.strip()
            else "Enter a question first."
        )

# ABOUT PROJECT
elif page == "About Project":
    st.subheader("ℹ️ About GraphMind AI")

    st.markdown("""
    ### Automatic Text-to-Knowledge Graph Generator

    **Technologies**
    - Python
    - Streamlit
    - spaCy
    - NetworkX
    - PyVis
    - PyMuPDF
    - Pandas

    **Innovations**
    - Entity normalization
    - Context-based relationship extraction
    - Confidence scoring
    - Evidence tracking
    - Interactive graph visualization
    - Graph-based question answering

    **Future Scope**
    - Semantic AI extraction
    - Contradiction detection
    - Multi-document knowledge fusion
    - Automated graph updating
    """)

# EXPORT
st.divider()
st.subheader("📥 Export Knowledge Graph")

c1,c2,c3 = st.columns(3)

with c1:
    data = json.dumps(nx.node_link_data(graph), indent=2)
    st.download_button(
        "Download JSON", data,
        "knowledge_graph.json", "application/json"
    )

with c2:
    st.download_button(
        "Download Entities CSV",
        entities.to_csv(index=False),
        "entities.csv", "text/csv"
    )

with c3:
    st.download_button(
        "Download Relationships CSV",
        relations.to_csv(index=False),
        "relationships.csv", "text/csv"
    )

# FOOTER
st.markdown("""
<div style="background:#172554;padding:20px;
border-radius:12px;text-align:center;margin-top:25px">
<h3 style="color:#c4b5fd">🧠 GraphMind AI</h3>
<p style="color:white">
Automatic Text-to-Knowledge Graph Generator
</p>
<p style="color:#cbd5e1">
Python | NLP | NetworkX | Streamlit | Version 3.0
</p>
</div>
""", unsafe_allow_html=True)