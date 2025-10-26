# ESN-unofficial

Indications for brain_connectome_reservoir.py

ConnectomeReservoir builds an ESN-style reservoir from one or more brain connectome graphs (GraphML). It:
1.	loads and merges graphs into a single adjacency,
2.	optionally resizes it,
3.	scales it to a target spectral radius,
4.	creates input weights and per-neuron leak rates,
5.	provides single-step and batched forward simulation utilities
- How the reservoir is built: check __init__ and humand_data.ipynb

•	it finds all GraphML files – reads all of them with NetworkX, takes union of node IDs and maps each graph onto a common node order. 
•	N.B: if edge_attr exists in at least one graph, and is numeric, it uses it, otherwise unweighted edges are used (but are suboptimal); if multiple graphs are provided, it combines their adjacencies by mean or median.
•	All matrices are symmetrized, and self loops are removed (zero on diagonal; not necessary, but faster in computation)
•	Spectral radius is estimated via power iteration
•	Win ∈ ℝ^{N×n_inputs} sampled uniform in [-input_scale, input_scale].
•	input_bias ∈ ℝ^{N} sampled same range (used for an implicit bias input of 1.0).
•	leak ∈ ℝ^{N} sampled uniform in [lo, hi] from leak_range.
•	Convenience aliases (for the input of the graphs; however it might slow down computation):
o	self.w = self.W
o	self.win = [Win | input_bias] (concatenates bias as an extra input column) so you can compute win @ [u; 1].
•	The internal states are initialized to zeroes
•	Forward-> accepts u as shape [T, n_inputs] or [[T], 1]
	Evolves using the formula in the function (classic forward formula for the reservoir, with the tanh)
	If collect_states=True, x ∈ ℝ^{TxN}
	If wout given (shape: [N] or [N, K]), it returns Y = X@wout
o	Returns:
	(X,Y) if both collect_states and wout are provided
	Y if only wout is provided
	X if only collect_states is TRUE
	None otherwise
•	Resize_adjancecy -> downsizes by keeping highest degree node
