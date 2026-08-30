"""
Re-evalua TODAS las versiones guardadas en model_versions/ contra el
held-out ACTUAL (el de hoy, no el que existia cuando se guardo cada version).

Util cuando el held-out crece: como cada version guarda los pesos reales
(cascade.xml + wally_cnn.pt), se les puede "tomar el examen de nuevo" en
cualquier momento y comparar todas en pie de igualdad, aunque en su momento
se hayan medido contra un held-out mas chico. Solo es justo si las escenas
del held-out actual son nuevas para TODAS las versiones (nunca se usaron
para entrenar ninguna) -- si agregaste una escena que ya estaba en el
training set de alguna version, esa version tiene ventaja injusta ahi.

Uso:
    python compare_versions.py
    python compare_versions.py --save-meta   # ademas guarda el resultado en el meta.json de cada version
"""

import argparse
import json
import os
from datetime import datetime, timezone

from cnn_reclassifier.pipeline import WallyDetector
from cnn_reclassifier.prepare_dataset import gather_wally_boxes, held_out_split
from evaluate import run, summarize
from save_version import VERSIONS_DIR, existing_versions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-meta", action="store_true", help="guardar el resultado en el meta.json de cada version")
    args = ap.parse_args()

    versions = existing_versions()
    if not versions:
        raise SystemExit(f"No hay versiones guardadas en {VERSIONS_DIR}.")

    all_boxes = gather_wally_boxes()
    _, test_boxes = held_out_split(all_boxes)
    print(f"Held-out actual: {len(test_boxes)} escenas -- {', '.join(sorted(os.path.basename(p) for p in test_boxes))}\n")

    results = []
    scene_names = sorted(os.path.basename(p) for p in test_boxes)
    for n, slug, vdir in versions:
        meta = json.load(open(os.path.join(vdir, "meta.json")))
        conf = meta.get("conf", 0.9998)
        cascade_path = os.path.join(vdir, "cascade.xml")
        model_path = os.path.join(vdir, "wally_cnn.pt")
        print(f"Evaluando v{n} ({slug})...")
        det = WallyDetector(cascade_path=cascade_path, model_path=model_path)
        rows = run(det, test_boxes, conf)
        summary = summarize(rows)
        results.append((n, slug, conf, summary, {r["name"]: r["hit"] for r in rows}))

        if args.save_meta:
            meta["held_out_recheck"] = dict(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                n_scenes=summary["n"],
                recall=summary["recall"],
                fp_total=summary["fp_total"],
            )
            json.dump(meta, open(os.path.join(vdir, "meta.json"), "w"), indent=2)

    # tabla resumen
    print(f"\n{'ver':<4} {'tag':<24} {'conf':>8} | {'recall':>10} {'FP':>5}")
    print("-" * 60)
    for n, slug, conf, summary, _ in results:
        print(f"v{n:<3} {slug:<24} {conf:>8} | {summary['hits']:>3}/{summary['n']:<3} ({summary['recall']*100:>3.0f}%) {summary['fp_total']:>5}")

    # matriz escena x version
    print(f"\n{'escena':<10}" + "".join(f"v{n:<5}" for n, *_ in results))
    for name in scene_names:
        row = f"{name:<10}"
        for _, _, _, _, hits in results:
            row += f"{'OK' if hits.get(name) else '-':<6}"
        print(row)


if __name__ == "__main__":
    main()
