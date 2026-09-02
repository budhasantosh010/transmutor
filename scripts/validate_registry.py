from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
rows=[]
for line in (root/'registry'/'experiments.jsonl').read_text().splitlines():
    if line.strip(): rows.append(json.loads(line))
ids=[r['id'] for r in rows]
assert len(ids)==len(set(ids)), 'duplicate experiment IDs'
valid={'RAW_SOURCE_PRESERVED','RESULT_ARTIFACT_PRESERVED','DOCUMENTED_IN_LOG','MISSING_DIRECT_ARTIFACT'}
assert all(r['provenance'] in valid for r in rows)
print(f'validated {len(rows)} experiment records')
