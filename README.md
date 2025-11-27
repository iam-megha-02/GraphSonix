# 🎼 GraphSonix  
### *Transforming Graph Embeddings Into Music*

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

GraphSonix converts **graph structures into music** using Node2Vec embeddings, random walks, and a custom audio synthesizer.  
Think: *graph theory meets music production*, but slightly unhinged.

---

## 🎵 Why GraphSonix?

Because your graphs deserve to sing.

GraphSonix blends graph properties into musical signals:

| Graph Concept | Music Mapping |
|---------------|--------------|
| Communities | Chords |
| Embeddings | Pitch |
| Betweenness | Duration |
| Degree | Volume |
| Random Walks | Melodies |

Topology never sounded so weirdly satisfying.

---

## Features

### Quick Node Player
Play nodes like sound samples.  
Every node = unique frequency, duration, volume.

### Custom Track Composer
- Add nodes (repeat allowed)  
- Collapsible node-details view  
- Total track duration limit  
- Live warnings when track overflows  
- Export full WAV audio  

### Upload & Visualize Graphs
- Accepts CSV, TXT, JSON  
- Automatic parsing  
- Interactive PyVis visualization  
- Node highlights & path colors  

### “Math Behind It”
Explained for humans, not mathematicians.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/graphsonix.git
cd graphsonix
pip install -r requirements.txt
streamlit run app.py
