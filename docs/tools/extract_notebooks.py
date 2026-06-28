"""
extract_notebooks.py makes lightweight, context-friendly extracts of the repo's
Jupyter notebooks (code and markdown cells, with image outputs stripped and long
text outputs truncated). The raw .ipynb files hold multi-MB of embedded base64
images, so reading them directly will blow up an LLM context window. Run this first,
then analyze the .txt extracts.

Usage (from the repo root):
    python docs/tools/extract_notebooks.py
Writes extracts to ./_nb_extracts/ (gitignored) and a _manifest.json summary.
"""
import json, os, re, glob, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
OUT = os.path.join(REPO, "_nb_extracts")
os.makedirs(OUT, exist_ok=True)


def cell_src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def short_outputs(cell):
    outs = []
    for o in cell.get("outputs", []) or []:
        ot = o.get("output_type")
        if ot == "stream":
            txt = o.get("text", "")
            outs.append(("stream", "".join(txt) if isinstance(txt, list) else txt))
        elif ot in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                txt = data["text/plain"]
                outs.append(("text", "".join(txt) if isinstance(txt, list) else txt))
            if [k for k in data if k.startswith("image/")]:
                outs.append(("image", "<image omitted>"))
        elif ot == "error":
            outs.append(("error", f"{o.get('ename','')}: {o.get('evalue','')}"))
    return outs


def main():
    manifest = []
    for path in sorted(glob.glob(os.path.join(REPO, "*.ipynb")) +
                       glob.glob(os.path.join(REPO, "**", "*.ipynb"), recursive=True)):
        name = os.path.relpath(path, REPO).replace(os.sep, "__")
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            manifest.append({"name": name, "error": str(e)})
            continue
        cells = nb.get("cells", [])
        code = "\n".join(cell_src(c) for c in cells if c.get("cell_type") == "code")
        imports = sorted({(a or b) for a, b in re.findall(
            r'(?m)^\s*(?:import\s+([\w\.]+)|from\s+([\w\.]+)\s+import)', code)})
        defs = re.findall(r'(?m)^\s*(?:def|class)\s+(\w+)', code)
        outpath = os.path.join(OUT, name.replace(".ipynb", ".txt"))
        with open(outpath, "w", encoding="utf-8") as g:
            g.write(f"# EXTRACT OF {name}  ({len(cells)} cells)\n")
            for i, c in enumerate(cells):
                ct = c.get("cell_type")
                if ct in ("markdown", "code"):
                    g.write(f"\n===== CELL {i} [{ct}] =====\n{cell_src(c)}\n")
                    if ct == "code":
                        for kind, txt in short_outputs(c):
                            t = txt if len(txt) < 1500 else txt[:1500] + "...<truncated>"
                            g.write(f"    [OUT {kind}] {t}\n")
        manifest.append({
            "name": name, "n_cells": len(cells),
            "code_chars": len(code), "imports": imports, "defs": defs,
            "code_hash": hashlib.md5(re.sub(r"\s+", "", code).encode()).hexdigest()[:10],
        })
    with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for m in manifest:
        print(m.get("name"), "->", m.get("code_hash", m.get("error")))
    print("\nExtracts written to", OUT)


if __name__ == "__main__":
    main()
