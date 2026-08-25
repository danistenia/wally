"""
Guarda una "foto" versionada del pipeline actual (cascade.xml + wally_cnn.pt)
para poder mostrar la evolucion del modelo (por ejemplo en un post) y volver
a correr detect_wally.py contra una version vieja para comparar.

Cada version queda en model_versions/vN_<slug>/ con una copia de los dos
archivos de modelo + meta.json (metricas de evaluate.py si coinciden con
esta version, commit de git, etc). Se commitea a git como cualquier otro
archivo -- son ~800KB por version, no hace falta Git LFS.

Uso:
    python evaluate.py                        # (opcional) medir antes de guardar
    python save_version.py --tag "round3-mining"
    python save_version.py --tag "mas-escenas" --note "sumadas 5 escenas nuevas"
    python save_version.py --list             # tabla de todas las versiones guardadas
"""

import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone

from cnn_reclassifier.pipeline import DEFAULT_CASCADE, DEFAULT_MODEL
from evaluate import LOG_PATH, git_info, model_fingerprint

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.join(SRC_DIR, "model_versions")


def existing_versions():
    """[(n, slug, dir_path), ...] ordenadas por n."""
    out = []
    for d in sorted(glob.glob(os.path.join(VERSIONS_DIR, "v*"))):
        name = os.path.basename(d)
        m = re.match(r"v(\d+)_(.+)", name)
        if m:
            out.append((int(m.group(1)), m.group(2), d))
    return sorted(out, key=lambda t: t[0])


def next_version_num():
    versions = existing_versions()
    return (versions[-1][0] + 1) if versions else 1


def find_matching_eval(fingerprint):
    """Ultima entrada de eval_runs.jsonl cuyo model_fingerprint coincida."""
    if not fingerprint or not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH) as f:
        lines = f.readlines()
    for line in reversed(lines):
        rec = json.loads(line)
        if rec.get("model_fingerprint") == fingerprint:
            return rec
    return None


def save(tag, note):
    fingerprint = model_fingerprint(DEFAULT_CASCADE, DEFAULT_MODEL)
    if fingerprint is None:
        raise SystemExit("No encuentro cascade.xml y/o wally_cnn.pt -- nada para guardar.")

    n = next_version_num()
    slug = re.sub(r"[^a-z0-9-]+", "-", tag.lower()).strip("-") or f"version{n}"
    version_dir = os.path.join(VERSIONS_DIR, f"v{n}_{slug}")
    os.makedirs(version_dir, exist_ok=True)

    shutil.copy2(DEFAULT_CASCADE, os.path.join(version_dir, "cascade.xml"))
    shutil.copy2(DEFAULT_MODEL, os.path.join(version_dir, "wally_cnn.pt"))

    commit, dirty = git_info()
    eval_rec = find_matching_eval(fingerprint)
    if eval_rec is None:
        print(
            "AVISO: no encontre una corrida en eval_runs.jsonl para el modelo actual.\n"
            "       Corre 'python evaluate.py' antes de guardar la version para que\n"
            "       quede con metricas reales adjuntas (si no, meta.json las deja en null)."
        )

    meta = dict(
        version=n,
        tag=slug,
        note=note,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        model_fingerprint=fingerprint,
        train=(eval_rec or {}).get("train"),
        test=(eval_rec or {}).get("test"),
        conf=(eval_rec or {}).get("conf"),
    )
    with open(os.path.join(version_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Guardado: {os.path.relpath(version_dir, SRC_DIR)}")
    if eval_rec:
        tr, te = eval_rec["train"], eval_rec["test"]
        tr_s = f"{tr['hits']}/{tr['n']}" if tr.get("n") else "-"
        te_s = f"{te['hits']}/{te['n']}" if te.get("n") else "-"
        print(f"  train: {tr_s} recall, {tr.get('fp_total','-')} FP | test: {te_s} recall, {te.get('fp_total','-')} FP")
    print(f"  no te olvides de: git add {os.path.relpath(version_dir, SRC_DIR)} && git commit")


def list_versions():
    versions = existing_versions()
    if not versions:
        print(f"No hay versiones guardadas todavia en {VERSIONS_DIR}.")
        return
    hdr = f"{'ver':<4} {'tag':<20} {'commit':<10} {'conf':>8} | {'TRAIN':>14} | {'TEST':>14}"
    print(hdr)
    print("-" * len(hdr))
    for n, slug, d in versions:
        meta_path = os.path.join(d, "meta.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        tr, te = meta.get("train") or {}, meta.get("test") or {}
        tr_s = f"{tr['hits']}/{tr['n']} ({tr.get('fp_total','-')} FP)" if tr.get("n") else "sin datos"
        te_s = f"{te['hits']}/{te['n']} ({te.get('fp_total','-')} FP)" if te.get("n") else "sin datos"
        commit = (meta.get("git_commit") or "?") + ("+" if meta.get("git_dirty") else "")
        conf = meta.get("conf", "-")
        print(f"v{n:<3} {slug:<20} {commit:<10} {conf:>8} | {tr_s:>14} | {te_s:>14}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="nombre corto de esta version, ej. 'round3-mining'")
    ap.add_argument("--note", default="", help="descripcion libre para el post")
    ap.add_argument("--list", action="store_true", help="listar las versiones ya guardadas")
    args = ap.parse_args()

    if args.list:
        list_versions()
        return
    if not args.tag:
        raise SystemExit("Falta --tag (o usa --list para ver las versiones guardadas)")
    save(args.tag, args.note)


if __name__ == "__main__":
    main()
