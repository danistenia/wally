"""
Evalua el modelo actual (cnn_reclassifier/wally_cnn.pt + cascade.xml) contra
una lista EXPLICITA de escenas de entrenamiento (no contra "todo lo que no
es held-out", que es lo que hace evaluate.py) y guarda la version en
model_versions/, igual que save_version.py pero con el TRAIN correcto para
un dataset restringido.

El TEST (held-out) se completa despues con:
    python compare_versions.py --save-meta

Uso:
    python save_replay_version.py --scenes 1,2,3,... --tag "replay-v1" --note "..."
"""

import argparse
import json
import os

from cnn_reclassifier.pipeline import DEFAULT_CASCADE, DEFAULT_MODEL, WallyDetector
from cnn_reclassifier.prepare_dataset import gather_wally_boxes
from evaluate import DEFAULT_CONF, git_info, model_fingerprint, run, summarize
from save_version import VERSIONS_DIR, next_version_num
import shutil
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="lista de stems separados por coma (el TRAIN historico exacto)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    all_boxes = gather_wally_boxes()
    train_boxes = {
        p: b for p, b in all_boxes.items()
        if os.path.splitext(os.path.basename(p))[0] in wanted
    }
    print(f"Evaluando TRAIN sobre {len(train_boxes)} escenas historicas...")
    det = WallyDetector()
    rows = run(det, train_boxes, args.conf)
    train_summary = summarize(rows)
    print(f"  TRAIN: {train_summary['hits']}/{train_summary['n']} recall, {train_summary['fp_total']} FP")

    fingerprint = model_fingerprint(DEFAULT_CASCADE, DEFAULT_MODEL)
    n = next_version_num()
    slug = args.tag
    version_dir = os.path.join(VERSIONS_DIR, f"v{n}_{slug}")
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy2(DEFAULT_CASCADE, os.path.join(version_dir, "cascade.xml"))
    shutil.copy2(DEFAULT_MODEL, os.path.join(version_dir, "wally_cnn.pt"))

    commit, dirty = git_info()
    meta = dict(
        version=n,
        tag=slug,
        note=args.note,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        model_fingerprint=fingerprint,
        conf=args.conf,
        train=train_summary,
        train_scenes=sorted(wanted, key=lambda x: (len(x), x)),
        test=None,
    )
    with open(os.path.join(version_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Guardado: {os.path.relpath(version_dir)}")
    print("  (falta el TEST -- correr: python compare_versions.py --save-meta)")


if __name__ == "__main__":
    main()
