"""
Compara corridas de evaluate.py guardadas en eval_runs.jsonl para saber si
una iteracion (mining, reentrenar, calibrar, cambiar --conf) mejoro o
empeoro, en vez de juzgar a ojo comparando imagenes sueltas.

Uso:
    cd src
    python compare_runs.py                 # ultima corrida vs. la anterior
    python compare_runs.py --baseline 3     # ultima corrida vs. 3 corridas atras
    python compare_runs.py --trend 5        # tabla de las ultimas 5 corridas (sin veredicto)
"""

import argparse
import json
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SRC_DIR, "eval_runs.jsonl")


def load_runs():
    if not os.path.exists(LOG_PATH):
        return []
    runs = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def label(run):
    commit = run.get("git_commit") or "sin-git"
    dirty = "+dirty" if run.get("git_dirty") else ""
    fp = run.get("model_fingerprint") or "?"
    return f"{run['timestamp']} ({commit}{dirty}, modelo {fp[:8]}, conf={run['conf']})"


def fmt_split(s):
    if not s or not s.get("n"):
        return "sin escenas"
    return f"{s['hits']}/{s['n']} (recall {s['recall']*100:.0f}%), {s['fp_total']} FP"


def verdict(before, after):
    b_n, a_n = (before or {}).get("n") or 0, (after or {}).get("n") or 0
    if not b_n or not a_n:
        return "SIN DATOS (falta escenas etiquetadas en alguna de las dos corridas)"
    r_before, r_after = before["recall"], after["recall"]
    if r_after > r_before:
        return "MEJORO (recall subio)"
    if r_after < r_before:
        return "EMPEORO (recall bajo)"
    fp_before, fp_after = before["fp_total"], after["fp_total"]
    delta = fp_after - fp_before
    grew_a_lot = delta >= 3 and (fp_before == 0 or delta > 0.2 * fp_before)
    if delta > 0 and grew_a_lot:
        return f"EMPEORO (mismo recall, +{delta} falsos positivos)"
    if delta < 0:
        return f"MEJORO (mismo recall, {delta} falsos positivos)"
    return "IGUAL"


def scene_key(row):
    return (row["split"], row["name"])


def scene_diffs(before_scenes, after_scenes):
    before = {scene_key(r): r for r in before_scenes}
    after = {scene_key(r): r for r in after_scenes}

    fixed, regressed, fp_changes, new_scenes = [], [], [], []
    for key, a in after.items():
        b = before.get(key)
        if b is None:
            new_scenes.append(key)
            continue
        if b["hit"] and not a["hit"]:
            regressed.append(key)
        elif not b["hit"] and a["hit"]:
            fixed.append(key)
        d = a["fp"] - b["fp"]
        if d != 0:
            fp_changes.append((key, b["fp"], a["fp"], d))
    return fixed, regressed, fp_changes, new_scenes


def compare(before, after):
    print(f"Antes:   {label(before)}")
    print(f"Despues: {label(after)}\n")

    for split, title in (("train", "TRAIN (in-sample)"), ("test", "TEST (held-out)")):
        b, a = before.get(split), after.get(split)
        print(f"--- {title} ---")
        print(f"  antes:   {fmt_split(b)}")
        print(f"  despues: {fmt_split(a)}")
        print(f"  veredicto: {verdict(b, a)}\n")

    fixed, regressed, fp_changes, new_scenes = scene_diffs(before["scenes"], after["scenes"])

    if regressed:
        print(f"Escenas que dejaron de encontrarse ({len(regressed)}):")
        for split, name in regressed:
            print(f"  - [{split}] {name}")
        print()
    if fixed:
        print(f"Escenas que ahora se encuentran y antes no ({len(fixed)}):")
        for split, name in fixed:
            print(f"  - [{split}] {name}")
        print()
    if fp_changes:
        fp_changes.sort(key=lambda t: abs(t[3]), reverse=True)
        print("Cambios en falsos positivos por escena (top 10 por magnitud):")
        for (split, name), b_fp, a_fp, d in fp_changes[:10]:
            signo = "+" if d > 0 else ""
            print(f"  - [{split}] {name}: {b_fp} -> {a_fp} ({signo}{d})")
        print()
    if new_scenes:
        print(f"Escenas nuevas en esta corrida (recien etiquetadas, sin comparacion previa): "
              f"{', '.join(name for _, name in new_scenes)}")
    if not (regressed or fixed or fp_changes or new_scenes):
        print("Sin cambios escena por escena.")


def print_trend(runs, k):
    runs = runs[-k:]
    hdr = f"{'timestamp':<25} {'commit':<10} {'conf':>8} | {'TRAIN recall':>13} {'TRAIN fp':>9} | {'TEST recall':>12} {'TEST fp':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        tr, te = r.get("train") or {}, r.get("test") or {}
        tr_r = f"{tr['hits']}/{tr['n']}" if tr.get("n") else "-"
        te_r = f"{te['hits']}/{te['n']}" if te.get("n") else "-"
        commit = (r.get("git_commit") or "?") + ("+" if r.get("git_dirty") else "")
        print(
            f"{r['timestamp']:<25} {commit:<10} {r['conf']:>8} | "
            f"{tr_r:>13} {tr.get('fp_total', '-'):>9} | {te_r:>12} {te.get('fp_total', '-'):>8}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=int, default=1,
                     help="comparar la ultima corrida contra N corridas atras (default 1 = la anterior)")
    ap.add_argument("--trend", type=int, default=None,
                     help="en vez de comparar dos corridas, listar las ultimas N en una tabla")
    args = ap.parse_args()

    runs = load_runs()
    if len(runs) < 1:
        raise SystemExit(f"No hay corridas guardadas todavia en {LOG_PATH}. Corre evaluate.py primero.")

    if args.trend:
        print_trend(runs, args.trend)
        return

    if len(runs) <= args.baseline:
        raise SystemExit(
            f"Solo hay {len(runs)} corrida(s) guardada(s), no alcanza para comparar "
            f"contra {args.baseline} atras. Corre evaluate.py de nuevo despues de tu proximo cambio."
        )

    compare(runs[-1 - args.baseline], runs[-1])


if __name__ == "__main__":
    main()
