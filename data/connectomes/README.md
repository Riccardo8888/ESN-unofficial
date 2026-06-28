# data/connectomes

A small committed subset of real structural connectomes, kept here so the connectome examples and the
null-model baseline (the null-model section of `examples/combined_examples.ipynb`) reproduce on a fresh clone.

`scale83/` holds 5 subjects in the 83-region Lausanne/Cammoun parcellation. The filenames carry the atlas
token `scale33`, which refers to the 83-region scale (a node count, not a file count). The graphmls are
undirected and weighted; the meaningful edge weight is `number_of_fibers` (streamline count). About 824 KB
in total.

**Source & terms.** These graphs are a subset of the **braingraph.org** "High-Resolution Structural
Connectomes" database (Budapest Reference Connectome project), computed from imaging data of the **WU-Minn
Human Connectome Project (HCP)**. They are redistributed here under the WU-Minn HCP Open Access Data Use
Terms (permitted by Term 4 for derived data). Before using them you must read and accept those terms,
include the HCP acknowledgement, and cite the braingraph.org papers — all reproduced in
[`DATA_TERMS.md`](DATA_TERMS.md). Do not attempt to identify or contact the subjects, and follow your
institution's human-subjects rules. This data is NOT under the repository's MIT license.

This is only a representative subset. The full multi-scale dataset (83 / 129 / 234 / 463 / 1015 nodes,
roughly 1000 subjects per scale) runs from hundreds of MB to several GB per scale, so it is kept external
and not committed. To use it, point `ConnectomeReservoir(graph_dir=...)` at a local copy. See
`docs/METHODOLOGY.md`.
