"""Check that adding S* (v3-7) changed nothing else: for every report of the new maps, the old report equals the new
one after removing R1.certified_saving and input.lock; S* in the report equals certsave.jsonl; map lines equal except
report_sha16 / lock fields.  python verify_v37.py OLD_MAP NEW_MAP CERTSAVE_JSONL CLAIM"""
import json, sys
from pathlib import Path
old, new, cs_path, claim = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
CS = {}
for l in open(cs_path):
    x = json.loads(l)
    if x["claim"] == claim:
        CS[(x["pool"], x["small"], x["big"], x["measure"])] = x["certified_saving"]
bad = n = scmp = 0
for f in sorted((new / "reports").rglob("*.json")):
    g = old / "reports" / f.relative_to(new / "reports")
    a, b = json.load(open(g)), json.load(open(f))
    cs = b["R1_verdict"].pop("certified_saving", None)
    a["input"].pop("lock", None); b["input"].pop("lock", None)
    n += 1
    if a != b:
        bad += 1
        if bad <= 3: print("DIFF", f.name)
    if "sensitivity" not in f.parts and cs is not None:
        k = tuple(f.stem.split("__"))
        if k in CS:
            scmp += 1
            if CS[k] != cs["value"]:
                bad += 1; print("S* differs", f.name, CS[k], cs["value"])
if not (new / "map.jsonl").exists():
    print(f"reports {n}, differing {bad}; S* compared {scmp}; map.jsonl not written yet"); sys.exit()
ol = {json.dumps({k: v for k, v in json.loads(l).items() if k not in ("report_sha16",)}, sort_keys=True) for l in open(old / "map.jsonl")}
nl = {json.dumps({k: v for k, v in json.loads(l).items() if k not in ("report_sha16",)}, sort_keys=True) for l in open(new / "map.jsonl")}
print(f"reports {n}, differing {bad}; S* compared {scmp}; map lines equal (except report hash): {ol == nl} ({len(nl)})")
# strict map comparison ignoring every report hash (top level and inside sensitivity cells)
def strip(x):
    x = dict(x); x.pop("report_sha16", None)
    if x.get("sensitivity"):
        x["sensitivity"] = {c: {k: v for k, v in r.items() if k != "report_sha16"} for c, r in x["sensitivity"].items()}
    return json.dumps(x, sort_keys=True)
print("map lines equal ignoring all report hashes:",
      {strip(json.loads(l)) for l in open(old / "map.jsonl")} == {strip(json.loads(l)) for l in open(new / "map.jsonl")})
