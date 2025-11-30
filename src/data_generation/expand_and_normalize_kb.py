"""Normalize, deduplicate and expand existing KB and QA data.
- Load `data/generated/knowledge_base.json` and `data/generated/medical_qa_dataset.json`.
- Normalize names, merge synonyms using simple lowercase/strip rules.
- Extract co-occurrence disease<->symptom and disease<->drug from QA dataset (question/explanation fields).
- Assign confidence scores: ontology=0.9, heuristic=0.6, cooccurrence=0.4, template_generated=0.3
- Save expanded KB to `data/generated/knowledge_base_expanded.json` and CSV.
"""
from pathlib import Path
import json
import csv
import re
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
GEN_DIR = DATA_DIR / 'generated'

KB_PATH = GEN_DIR / 'knowledge_base.json'
QA_PATH = GEN_DIR / 'medical_qa_dataset.json'
OUT_JSON = GEN_DIR / 'knowledge_base_expanded.json'
OUT_CSV = GEN_DIR / 'knowledge_base_expanded.csv'


def load_json(p: Path):
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize(s: str) -> str:
    if not s:
        return ''
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('\u200b','')
    return s


def keyify(s: str) -> str:
    s = normalize(s).lower()
    s = re.sub(r'[^\w\s\-]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


kb = load_json(KB_PATH)
qa = load_json(QA_PATH).get('data') if isinstance(load_json(QA_PATH), dict) else load_json(QA_PATH)

# 1) Normalize existing KB and deduplicate
entries = {}
for row in kb:
    d = normalize(row.get('disease',''))
    de = normalize(row.get('disease_en',''))
    ent = normalize(row.get('entity',''))
    ent_en = normalize(row.get('entity_en',''))
    etype = row.get('entity_type','') or ''
    rel = row.get('relation','') or ''
    source_type = row.get('source_type','') or ''
    source_name = row.get('source_name','') or ''
    src_id = row.get('source_id','') or ''
    k = (keyify(d), keyify(ent), etype, rel)
    if k not in entries:
        entries[k] = {
            'disease': d,
            'disease_en': de,
            'entity': ent,
            'entity_en': ent_en,
            'entity_type': etype,
            'relation': rel,
            'source_type': source_type,
            'source_name': source_name,
            'source_id': src_id,
            'confidence': 0.6 if source_type=='heuristic' else 0.9 if source_type=='ontology' else 0.5
        }

# 2) Extract co-occurrence from QA dataset
# Heuristic: if a QA item mentions a disease name and a symptom/drug term, create relation
# First build term lists
all_diseases = set([e['disease'] for e in entries.values() if e['disease']])
all_entities = set([e['entity'] for e in entries.values() if e['entity']])

# also expand from QA: collect frequent tokens that look like disease or symptom
qa_texts = []
if qa:
    for item in qa:
        txt = ' '.join([str(item.get('question','')), str(item.get('explanation',''))])
        qa_texts.append(txt)

co_counts = Counter()
for txt in qa_texts:
    lower = txt.lower()
    for d in all_diseases:
        if d and d.lower() in lower:
            for ent in all_entities:
                if ent and ent.lower() in lower and ent.lower()!=d.lower():
                    # decide relation: if entity type is 'drug' -> treated_by else has_symptom
                    # find entity type
                    ent_type = None
                    for e in entries.values():
                        if keyify(e['entity'])==keyify(ent):
                            ent_type = e['entity_type']
                            break
                    rel = 'treated_by' if ent_type=='drug' else 'has_symptom'
                    k = (keyify(d), keyify(ent), rel)
                    co_counts[k]+=1

# Add co-occurrence edges with low confidence
for (kd, ke, rel), cnt in co_counts.items():
    key = (kd, ke, '', rel)
    # map back to readable names
    disease = ''
    entity = ''
    for e in entries.values():
        if keyify(e['disease'])==kd:
            disease = e['disease']
            break
    for e in entries.values():
        if keyify(e['entity'])==ke:
            entity = e['entity']
            break
    if disease and entity:
        k = (kd, ke, rel)
        if k not in entries:
            entries[k] = {
                'disease': disease,
                'disease_en': '',
                'entity': entity,
                'entity_en': '',
                'entity_type': 'drug' if any(e['entity']==entity and e['entity_type']=='drug' for e in entries.values()) else 'symptom',
                'relation': rel,
                'source_type': 'cooccurrence',
                'source_name': 'qa_corpus',
                'source_id': '',
                'confidence': 0.4
            }

# 3) Simple expansion: for each disease, pair with common symptoms by template to multiply dataset (synthetic augmentation)
# Use symptom list from entries
symptoms = [e['entity'] for e in entries.values() if e['entity_type']=='symptom']

# We'll generate synthetic pairs but mark confidence low
synthetic_added = 0
for e in list(entries.values()):
    if e['disease'] and not e['entity']:
        # attach up to 3 symptoms randomly chosen
        import random
        choices = random.sample(symptoms, min(3, len(symptoms))) if symptoms else []
        for s in choices:
            k = (keyify(e['disease']), keyify(s), 'has_symptom')
            if k not in entries:
                entries[k] = {
                    'disease': e['disease'],
                    'disease_en': e.get('disease_en',''),
                    'entity': s,
                    'entity_en': '',
                    'entity_type': 'symptom',
                    'relation': 'has_symptom',
                    'source_type': 'synthetic',
                    'source_name': 'template_augment',
                    'source_id': '',
                    'confidence': 0.3
                }
                synthetic_added += 1

# 4) Finalize and write out
out = list(entries.values())

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fieldnames = ['disease','disease_en','entity','entity_en','entity_type','relation','source_type','source_name','source_id','confidence']
with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in out:
        writer.writerow({k: r.get(k,'') for k in fieldnames})

print(f'Wrote expanded KB with {len(out)} records (including {synthetic_added} synthetic) to {OUT_JSON} and {OUT_CSV}')
