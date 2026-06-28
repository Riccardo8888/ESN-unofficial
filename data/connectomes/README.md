# data/connectomes

A small, committed subset of real structural connectomes so the connectome examples + the
null-model baseline (`examples/07`) are reproducible on a fresh clone.

- **`scale83/`** — 5 subjects from the HCP dataset, **83-region Lausanne/Cammoun parcellation**
  (the filenames carry the atlas token `scale33`, which denotes the 83-region scale — node count, not file count).
  Undirected, weighted graphmls; the meaningful edge weight is `number_of_fibers` (streamline count).
  ~824 KB total.

This is a **representative subset**. The full multi-scale dataset (83 / 129 / 234 / 463 / 1015 nodes,
~1000 subjects per scale) is hundreds of MB to several GB per scale and is kept external (not committed).
To use it, point `ConnectomeReservoir(graph_dir=...)` at a local copy. See `docs/METHODOLOGY.md`.
