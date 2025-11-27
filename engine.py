import json
import io
import struct
import math
import pandas as pd

import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import json


SAMPLE_RATE = 44100


# ------------------------------------------------------------
# 1. LOAD GRAPH
# ------------------------------------------------------------
def load_graph(uploaded):
    name = uploaded.name.lower()

    try:
        # ---------------------------------------------------------
        # CSV HANDLING
        # ---------------------------------------------------------
        if name.endswith(".csv"):
            uploaded.seek(0)

            try:
                # Decode if bytes-like
                content = uploaded.read()
                text = content.decode("utf-8", errors="ignore")
                df = pd.read_csv(io.StringIO(text))
            except Exception as e:
                print("CSV read error:", e)
                return None

            # Case 1: correct column names
            if {"source", "target"} <= set(df.columns):
                return nx.from_pandas_edgelist(df, "source", "target")

            # Case 2: first two columns are edges
            if df.shape[1] >= 2:
                return nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])

            return None

        # ---------------------------------------------------------
        # TXT / EDGELIST
        # ---------------------------------------------------------
        if name.endswith(".txt") or name.endswith(".edgelist"):
            uploaded.seek(0)
            text = uploaded.read().decode("utf-8", errors="ignore")
            lines = text.strip().split("\n")

            edges = []
            for line in lines:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    edges.append((parts[0], parts[1]))

            G = nx.Graph()
            G.add_edges_from(edges)
            return G if G.number_of_nodes() > 0 else None

        # ---------------------------------------------------------
        # JSON
        # Format: { "edges": [[u,v], ...] }
        # ---------------------------------------------------------
        if name.endswith(".json"):
            uploaded.seek(0)
            try:
                data = json.load(uploaded)
            except:
                return None

            if "edges" in data:
                G = nx.Graph()
                G.add_edges_from(data["edges"])
                return G

            return None

        return None

    except Exception as e:
        print("Error:", e)
        return None


# ------------------------------------------------------------
# 2. NODE2VEC RANDOM WALKS
# ------------------------------------------------------------
def _node2vec_transition_probs(G, prev, curr, p=1.0, q=1.0):
    """
    Compute unnormalized transition probabilities from curr to its neighbors,
    given previous node 'prev', following Node2Vec rules.
    """
    neighbors = list(G.neighbors(curr))
    probs = []

    for x in neighbors:
        if x == prev:
            weight = 1.0 / p
        elif G.has_edge(x, prev):
            weight = 1.0
        else:
            weight = 1.0 / q
        probs.append(weight)

    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum() if probs.sum() > 0 else 1.0
    return neighbors, probs


def _node2vec_walk(G, start, walk_length=20, p=1.0, q=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    walk = [start]
    neighbors = list(G.neighbors(start))
    if not neighbors:
        return walk

    # first step: unbiased
    current = rng.choice(neighbors)
    walk.append(current)

    while len(walk) < walk_length:
        prev = walk[-2]
        curr = walk[-1]
        nbrs = list(G.neighbors(curr))
        if not nbrs:
            break

        nbrs, probs = _node2vec_transition_probs(G, prev, curr, p, q)
        nxt = rng.choice(nbrs, p=probs)
        walk.append(nxt)

    return walk


def embed_graph(G, dimensions=32, num_walks=20, walk_length=20, p=1.0, q=1.0):
    """
    Node2Vec-style embeddings implemented with Word2Vec.
    Returns: dict[node] -> embedding vector (np.ndarray)
    """
    rng = np.random.default_rng(42)
    nodes = list(G.nodes())
    walks = []

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for n in nodes:
            walk = _node2vec_walk(G, n, walk_length, p, q, rng)
            walks.append([str(v) for v in walk])

    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=5,
        min_count=0,
        sg=1,
        workers=1,
        epochs=10,
    )

    emb = {n: model.wv[str(n)] for n in G.nodes()}
    return emb


# ------------------------------------------------------------
# 3. GRAPH FEATURES
# ------------------------------------------------------------
def compute_graph_features(G, emb):
    # communities
    comms = nx.algorithms.community.greedy_modularity_communities(G)
    node_to_comm = {}
    for i, c in enumerate(comms):
        for n in c:
            node_to_comm[n] = i

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)

    return {
        "node_to_comm": node_to_comm,
        "degree": degree,
        "betweenness": betweenness,
    }


# ------------------------------------------------------------
# 4. EMBEDDING → NOTE EVENT
# ------------------------------------------------------------
def node_to_note_event(node, emb, features):
    """
    Map a node to (frequency, duration, volume)
    using its embedding, degree and betweenness.
    """
    vec = emb[node]
    degree = features["degree"][node]
    bet = features["betweenness"][node]

    # frequency: embedding norm projected and bounded
    proj = float(np.dot(vec, np.ones_like(vec)))
    base_freqs = [220, 247, 262, 294, 330, 349, 392, 440]  # A minor-ish scale
    idx = int(abs(proj)) % len(base_freqs)
    freq = base_freqs[idx]

    # duration: 220ms–520ms from betweenness
    duration = 0.22 + 0.3 * bet

    # volume: 0.4–0.9 from degree
    max_deg = max(features["degree"].values()) or 1
    vol = 0.4 + 0.5 * (degree / max_deg)

    return freq, duration, vol


# ------------------------------------------------------------
# 5. SYNTHESIZER
# ------------------------------------------------------------
def synth(freq, duration, volume):
    """
    Simple clean tone with mild harmonics + ADSR envelope.
    Returns float32 array in [-1, 1].
    """
    sr = SAMPLE_RATE
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # tri-harmonic tone
    base = np.sin(2 * math.pi * freq * t)
    h2 = 0.4 * np.sin(2 * math.pi * 2 * freq * t)
    h3 = 0.25 * np.sin(2 * math.pi * 3 * freq * t)
    wave = (base + h2 + h3) / 1.65

    # attack–release envelope
    attack = int(0.015 * sr)
    release = int(0.03 * sr)
    env = np.ones_like(wave)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env[-release:] = np.linspace(1.0, 0.0, release)

    wave *= env
    wave *= volume

    # soft clip
    wave = np.tanh(1.3 * wave)

    return wave.astype(np.float32)


# ------------------------------------------------------------
# 6. TRACK GENERATION
# ------------------------------------------------------------
def generate_track_from_nodes(path, emb, features):
    if not path:
        return None
    audio_segments = []
    for n in path:
        freq, dur, vol = node_to_note_event(n, emb, features)
        audio_segments.append(synth(freq, dur, vol))
    return np.concatenate(audio_segments)


# ------------------------------------------------------------
# 7. WAV ENCODING
# ------------------------------------------------------------
def make_wav(audio, sample_rate=SAMPLE_RATE):
    """
    audio: float32 array in [-1, 1]
    returns: bytes (WAV) for st.audio
    """
    if audio is None or len(audio) == 0:
        return None

    # clip and convert to int16
    a = np.clip(audio, -1.0, 1.0)
    int16 = (a * 32767).astype("<i2")

    buf = io.BytesIO()

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + int16.nbytes))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", int16.nbytes))
    buf.write(int16.tobytes())

    buf.seek(0)
    return buf.read()
