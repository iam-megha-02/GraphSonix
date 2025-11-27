import streamlit as st
import numpy as np
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

from engine import (
    load_graph,
    embed_graph,
    compute_graph_features,
    generate_track_from_nodes,
    node_to_note_event,
    synth,
    make_wav,
)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="GraphSonix",
    page_icon="🎼",
    layout="wide",
)


# ------------------------------------------------------------
# CSS (original dark theme)
# ------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #f0f6fc !important;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f0f6fc !important;
}
.big-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 46px;
    font-weight: 600;
    margin-bottom: 4px;
}
.app-sub {
    font-size: 18px;
    font-weight: 400;
    color: #8b949e;
    margin-bottom: 24px;
}
.section-card {
    background: #161b22;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #30363d;
    margin-bottom: 24px;
}

.section-card p, 
.section-card div, 
.section-card {
    color: #d7dce1 !important;   /* bright grey */
}


[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] * {
    color: #f0f6fc !important;
}
.stButton>button {
    background-color: #3b82f6 !important;
    color: white !important;
    padding: 0.55rem 1.2rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: none !important;
}
.stButton>button:hover {
    background-color: #1d4ed8 !important;
}
.stFileUploader {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    padding: 14px !important;
    border-radius: 10px !important;
}


.stFileUploader label {
    color: #f0f6fc !important;      /* bright white-ish */
    font-weight: 500 !important;
}


.stAudio {
    background-color: #161b22 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

h3 {
    color: #6b7a94 !important;   /* Light gray-blue visible on dark background */
    font-weight: 600 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "",
    [
        "Home",
        "Upload Graph",
        "Studio",        
        "Math Behind It",
    ],
)



# ------------------------------------------------------------
# PYVIS GRAPH
# ------------------------------------------------------------
def render_interactive_graph(
    G,
    features,
    height="500px",
    highlight_node=None,
    path=None,
):

    node_to_comm = features["node_to_comm"]
    degree = features["degree"]
    palette = [
        "#3b82f6", "#ec4899", "#22c55e", "#eab308",
        "#a855f7", "#f97316", "#14b8a6", "#ef4444"
    ]
    pathset = set(path or [])

    net = Network(height=height, width="100%", bgcolor="#0d1117", font_color="#fff")
    net.barnes_hut()

    # ---------- Node size controls ----------
    BASE_NODE_SIZE = 20
    DEGREE_MULTIPLIER = 25

    # ---------- Edge thickness controls ----------
    BASE_EDGE_WIDTH = 2
    HIGHLIGHT_EDGE_WIDTH = 5

    # ---------- Node label size ----------
    LABEL_FONT_SIZE = 28

    # ---------------------------------------
    # NODES
    # ---------------------------------------
    for n in G.nodes():
        comm = node_to_comm[n]
        color = palette[comm % len(palette)]

        size = BASE_NODE_SIZE + DEGREE_MULTIPLIER * (degree[n] / (max(degree.values()) or 1))
        border = 1

        if n in pathset:
            color = "#f97316"
            border = 3
        if highlight_node == n:
            color = "#eab308"
            border = 4
            size += 8

        net.add_node(
            n,
            label=str(n),
            color=color,
            size=size,
            borderWidth=border,
            font={'size': 28, 'color': '#ffffff', 'face': 'arial'}  # << Increase label size
        )


    # ---------------------------------------
    # EDGES
    # ---------------------------------------
    for u, v in G.edges():
        edge_color = "#6b7280"
        width = BASE_EDGE_WIDTH

        if u in pathset and v in pathset:
            edge_color = "#f97316"
            width = HIGHLIGHT_EDGE_WIDTH

        net.add_edge(u, v, color=edge_color, width=width)

    html_path = "graphsonix_temp.html"
    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=520)



# ------------------------------------------------------------
# PAGE: HOME  (About + Karate Demo)
# ------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="big-title">GraphSonix</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Where graphs learn to sing</div>', unsafe_allow_html=True)

    # ABOUT
    st.markdown(
        """
    <div class="section-card">
        <h3>About</h3>
        GraphSonix transforms graph embeddings into music.
        Node2Vec learns a vector for each node; communities become chords,
        random walks become melodies, and centrality shapes dynamics.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KARATE DEMO
    with st.expander("✨ Try a quick Karate Club demo"):
        G = nx.karate_club_graph()

        if "home_emb" not in st.session_state:
            st.session_state.home_emb = embed_graph(G)
            st.session_state.home_features = compute_graph_features(G, st.session_state.home_emb)

            start = max(G.degree, key=lambda x: x[1])[0]
            path = [start]
            cur = start
            for _ in range(24):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = np.random.choice(nbrs)
                path.append(cur)
            st.session_state.home_path = path

        if st.button("Refresh Demo Pattern"):
            st.session_state.home_emb = embed_graph(G)
            st.session_state.home_features = compute_graph_features(G, st.session_state.home_emb)

            start = max(G.degree, key=lambda x: x[1])[0]
            path = [start]
            cur = start
            for _ in range(24):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = np.random.choice(nbrs)
                path.append(cur)
            st.session_state.home_path = path

        emb = st.session_state.home_emb
        features = st.session_state.home_features
        path = st.session_state.home_path

        render_interactive_graph(G, features, path=path)
        audio = generate_track_from_nodes(path, emb, features)
        if audio is not None:
            st.audio(make_wav(audio))


# ------------------------------------------------------------
# PAGE: UPLOAD GRAPH
# ------------------------------------------------------------
elif page == "Upload Graph":
    st.markdown('<div class="big-title">Upload Your Graph</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload graph file", type=["csv", "txt", "edgelist", "json"])

    if uploaded:
        G = load_graph(uploaded)
        if G is None:
            st.error("Could not load graph.")
        else:
            st.success(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

            st.subheader("Original Graph Preview")
            render_interactive_graph(G, {"node_to_comm": {n:0 for n in G.nodes()}, "degree": dict(G.degree()), "betweenness": {n:0 for n in G.nodes()}})

            if st.button("Compute Embeddings"):
                st.session_state.G = G
                st.session_state.emb = embed_graph(G)
                st.session_state.features = compute_graph_features(G, st.session_state.emb)
                st.success("Embeddings computed!")

            if "features" in st.session_state and st.session_state.G is G:
                st.subheader("Embedded Graph View")
                render_interactive_graph(G, st.session_state.features)



# ------------------------------------------------------------
# PAGE: STUDIO (Node Player + Custom Track)
# ------------------------------------------------------------
elif page == "Studio":
    st.markdown('<div class="big-title">Studio</div>', unsafe_allow_html=True)
    if "emb" not in st.session_state:
        st.error("Upload a graph and compute embeddings first.")
    else:
        st.markdown('<div class="big-title">Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Play individual nodes or compose custom tracks</div>', unsafe_allow_html=True)

        G = st.session_state.G
        emb = st.session_state.emb
        features = st.session_state.features

        tab1, tab2 = st.tabs(["🎧 Quick Node Player", "🎼 Custom Track Composer"])


        # ----------------------------------------------------
        # TAB 1 — Quick Node Player with Pagination
        # ----------------------------------------------------
        with tab1:

            st.markdown("### Play Nodes Quickly")

            nodes_list = list(G.nodes())
            nodes_list.sort()

            page_size = 9  # 3 columns x 3 rows
            total_pages = max(1, (len(nodes_list) - 1) // page_size + 1)

            if "studio_page" not in st.session_state:
                st.session_state.studio_page = 1

            current = st.session_state.studio_page

            # Slice nodes for current page
            start = (st.session_state.studio_page - 1) * page_size
            end = start + page_size
            nodes_to_display = nodes_list[start:end]


            # ------------------------------------------
            # 3-COLUMN GRID
            # ------------------------------------------
            cols = st.columns(3)

            for idx, node in enumerate(nodes_to_display):
                with cols[idx % 3]:
                    freq, dur, vol = node_to_note_event(node, emb, features)

                    st.markdown(
                        f"""
                        <div style="
                            background:#161b22;
                            padding:14px;
                            border-radius:10px;
                            border:1px solid #30363d;
                            margin-bottom:10px;
                            min-height:120px;
                        ">
                            <div style="font-size:18px;font-weight:600;color:#f0f6fc;">Node {node}</div>
                            <div style="font-size:14px;color:#9ca3af;">Pitch: {freq} Hz</div>
                            <div style="font-size:14px;color:#9ca3af;">Duration: {dur:.2f}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</br>", unsafe_allow_html=True)
                    audio_bytes = make_wav(synth(freq, dur, vol))
                    st.audio(audio_bytes, format="audio/wav")
            
            # -------------------------------
            # Minimal Pagination (Working + Clean)
            # -------------------------------

            # CSS to make buttons look like small inline text links
            st.markdown("""
            <style>
            .small-btn > button {
                background: none !important;
                border: none !important;
                color: #3b82f6 !important;
                padding: 2px 6px !important;
                margin: 0 3px !important;
                font-size: 14px !important;
            }
            .small-btn > button:hover {
                text-decoration: underline !important;
            }
            .small-btn-active > button {
                background: none !important;
                border: none !important;
                color: #111 !important;
                font-weight: 600 !important;
                text-decoration: underline !important;
                padding: 2px 6px !important;
                margin: 0 3px !important;
                font-size: 14px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Working pagination logic
            page_size = 9
            nodes_list = list(G.nodes())
            nodes_list.sort()
            total_pages = max(1, (len(nodes_list) - 1) // page_size + 1)

            if "studio_page" not in st.session_state:
                st.session_state.studio_page = 1

            current_page = st.session_state.studio_page

            # Layout: <<  <   1 2 3 4 5   >   >>
            cols = st.columns(total_pages + 4)

            # First
            with cols[0]:
                if st.button("≪", key="first", help="First page"):
                    st.session_state.studio_page = 1

            # Prev
            with cols[1]:
                if st.button("‹", key="prev", help="Previous page"):
                    st.session_state.studio_page = max(1, current_page - 1)

            # Page numbers
            for i in range(1, total_pages + 1):
                css_class = "small-btn-active" if i == current_page else "small-btn"
                with cols[i + 1]:
                    if st.button(str(i), key=f"page_{i}", help=f"Go to page {i}", type="secondary"):
                        st.session_state.studio_page = i

            # Next
            with cols[total_pages + 2]:
                if st.button("›", key="next", help="Next page"):
                    st.session_state.studio_page = min(total_pages, current_page + 1)

            # Last
            with cols[total_pages + 3]:
                if st.button("≫", key="last", help="Last page"):
                    st.session_state.studio_page = total_pages

            # Slice nodes
            start = (st.session_state.studio_page - 1) * page_size
            end = start + page_size
            nodes_to_display = nodes_list[start:end]


        # ----------------------------------------------------
        # TAB 2 — Custom Track Composer (Enhanced Version)
        # ----------------------------------------------------
        st.markdown("""
        <style>
        /* Reduce overall vertical spacing inside this tab */
        .block-container {
            padding-top: 0rem !important;
        }

        /* Reduce spacing between elements */
        .stColumn {
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Reduce spacing between form fields */
        div[data-testid="stVerticalBlock"] {
            gap: 0.2rem !important;
        }

        /* Tighten selectbox and button area */
        .selectbox-row {
            margin-bottom: 0.5rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        with tab2:
            st.markdown("<h3 style='color:#e2e8f0;'>Custom Track Composer</h3>", unsafe_allow_html=True)

            nodes = list(G.nodes())
            nodes.sort()

            # ---------------------------------------
            # Track duration limit input
            # ---------------------------------------
            max_track_duration = st.number_input(
                "Set Maximum Track Duration (seconds)",
                min_value=1.0, max_value=60.0, value=10.0, step=1.0
            )
            # Maintain selected list
            if "selected_nodes" not in st.session_state:
                st.session_state.selected_nodes = []

            st.markdown("<h4 style='color:#94a3b8;'>Add Nodes to Track</h4>", unsafe_allow_html=True)

            # ---------------------------------------
            # MULTI-SELECTION with duplicates allowed
            # ---------------------------------------
            colA, colB = st.columns([2, 1])

            with colA:
                chosen = st.selectbox("Choose a node", nodes)

            with colB:
                st.markdown("<div style='height:38px;'></div>", unsafe_allow_html=True)
                if st.button("Add"):
                    # compute audio duration of chosen node
                    _, dur, _ = node_to_note_event(chosen, emb, features)

                    # compute current total duration
                    current_total = sum(node_to_note_event(n, emb, features)[1] 
                                        for n in st.session_state.selected_nodes)

                    if current_total + dur > max_track_duration:
                        st.error("Track duration limit reached. Cannot add more nodes.")
                    else:
                        st.session_state.selected_nodes.append(chosen)

            # ---------------------------------------
            # COLLAPSIBLE NODE DETAILS
            # ---------------------------------------
            total_duration = sum(node_to_note_event(n, emb, features)[1] 
                                for n in st.session_state.selected_nodes)

            with st.expander(f"Node Details (Total duration: {total_duration:.2f}s / {max_track_duration:.2f}s)"):
                for idx, n in enumerate(st.session_state.selected_nodes):
                    freq, dur, vol = node_to_note_event(n, emb, features)
                    st.markdown(
                        f"**{idx+1}. Node {n} → Pitch: {freq} Hz, Duration: {dur:.2f}s, Volume: {vol:.2f}**"
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------
            # Generate Track
            # ---------------------------------------
            if st.button("Create Track"):
                if not st.session_state.selected_nodes:
                    st.warning("Add nodes first.")
                else:
                    audio = generate_track_from_nodes(
                        st.session_state.selected_nodes, emb, features
                    )
                    st.audio(make_wav(audio))


elif page == "Math Behind It":
    st.markdown('<div class="big-title">Math Behind GraphSonix</div>', unsafe_allow_html=True)

    with st.expander("What are embeddings?"):
        st.write("""
Embeddings are like giving every node a “GPS coordinate”.
Nodes that are related or connected end up with similar coordinates.
""")

    with st.expander("How do we get these coordinates?"):
        st.write("""
We take random walks on the graph (like wandering through the network).
Nodes that appear in similar walks learn similar vector positions.
This is the idea behind Node2Vec and Word2Vec.
""")

    with st.expander("How does this become music?"):
        st.write("""
• The coordinate of a node decides its note (pitch).  
• A node's importance (centrality) decides how long the note plays.  
• A node's connections (degree) decide how loud it is.  
• A sequence of nodes becomes a melody.
""")

    with st.expander("Why does each track sound different?"):
        st.write("""
Because random walks explore different paths through the network,
each refresh gives a new melody that reflects the graph's structure.
""")


