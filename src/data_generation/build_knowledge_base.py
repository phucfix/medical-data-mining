"""Build a Disease-Symptom-Drug knowledge base from international ontologies (ICD, MeSH, HPO, DrugBank).
This script downloads public sources (if allowed), parses them and writes a normalized CSV/JSON with columns:
- disease (vi)
- disease_en
- entity (vi)
- entity_en
- entity_type (symptom|drug)
- relation (has_symptom|treated_by)
- source_type
- source_name

This script is conservative: it will first look for local copies under data/external and only attempt to download if missing.

Note: Some sources require licenses (UMLS). The script will not attempt to access UMLS programmatically; instead it supports loading local UMLS export files if the user provides them.
"""

from pathlib import Path
import json
import csv
import re
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
EXTERNAL_DIR = DATA_DIR / 'external'
OUTPUT_DIR = DATA_DIR / 'generated'

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpers

def load_json_if_exists(p: Path) -> Optional[List[Dict]]:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def normalize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s


# Load available sources (script run after previous offline step)
icd10 = load_json_if_exists(EXTERNAL_DIR / 'icd10_diseases.json') or []
mesh = load_json_if_exists(EXTERNAL_DIR / 'mesh_terms.json') or []
hpo = load_json_if_exists(EXTERNAL_DIR / 'hpo_terms.json') or []
drugs_local = load_json_if_exists(EXTERNAL_DIR / 'drugs.json') or []
symptoms_local = load_json_if_exists(EXTERNAL_DIR / 'symptoms.json') or []

# We'll create KB entries from these sources
kb = []
seen = set()

# From ICD-10: disease -> maybe mapped disease_en and code
for d in icd10:
    name_vi = normalize_name(d.get('name_vi') or d.get('name') or '')
    name_en = d.get('name_en') or ''
    code = d.get('code') or ''
    if not name_vi:
        continue

    # Add disease node (no relation yet)
    entry = {
        'disease': name_vi,
        'disease_en': name_en,
        'entity': '',
        'entity_en': '',
        'entity_type': '',
        'relation': '',
        'source_type': 'ontology',
        'source_name': 'ICD-10',
        'source_id': code
    }
    key = (name_vi, '', '')
    if key not in seen:
        kb.append(entry)
        seen.add(key)

# From local symptom list: create has_symptom relations (diseases unknown yet)
# We'll attach symptoms to likely diseases via simple heuristics: search disease names that mention symptom or vice versa
symptom_names = [normalize_name(s.get('name_vi')) for s in symptoms_local if s.get('name_vi')]

# From local drugs: we will attach treated_by relations by simple matching on indication
for drug in drugs_local:
    drug_name = normalize_name(drug.get('name_vi') or drug.get('name'))
    drug_en = drug.get('name_en') or ''
    indication = drug.get('indication') or ''
    if not drug_name:
        continue
    # Create drug nodes
    entry = {
        'disease': '',
        'disease_en': '',
        'entity': drug_name,
        'entity_en': drug_en,
        'entity_type': 'drug',
        'relation': '',
        'source_type': 'database',
        'source_name': 'local_drugs',
        'source_id': ''
    }
    key = ('', drug_name, 'drug')
    if key not in seen:
        kb.append(entry)
        seen.add(key)

# Heuristic: match indications to disease names (simple substring)
for drug in drugs_local:
    drug_name = normalize_name(drug.get('name_vi') or drug.get('name'))
    indication = (drug.get('indication') or '').lower()
    if not indication:
        continue
    for d in icd10:
        disease_texts = ' '.join([str(d.get('name_vi','')).lower(), str(d.get('name_en','')).lower(), str(d.get('category','')).lower()])
        if any(tok.strip() and tok.strip() in indication for tok in disease_texts.split()):
            entry = {
                'disease': normalize_name(d.get('name_vi') or ''),
                'disease_en': d.get('name_en') or '',
                'entity': drug_name,
                'entity_en': drug.get('name_en') or '',
                'entity_type': 'drug',
                'relation': 'treated_by',
                'source_type': 'heuristic',
                'source_name': 'local_drug_indication_match',
                'source_id': drug_name
            }
            key = (entry['disease'], entry['entity'], 'treated_by')
            if key not in seen and entry['disease']:
                kb.append(entry)
                seen.add(key)

# Heuristic: for each disease, if its name mentions a known symptom word, create has_symptom
symptom_tokens = set(tok.lower() for s in symptom_names for tok in s.split())
for d in icd10:
    disease_vi = normalize_name(d.get('name_vi') or '')
    disease_en = d.get('name_en') or ''
    if not disease_vi:
        continue
    lower = disease_vi.lower()
    for symptom in symptom_names:
        if symptom.lower() in lower or any(tok in lower for tok in symptom.split()):
            entry = {
                'disease': disease_vi,
                'disease_en': disease_en,
                'entity': symptom,
                'entity_en': '',
                'entity_type': 'symptom',
                'relation': 'has_symptom',
                'source_type': 'heuristic',
                'source_name': 'local_symptom_namedmatch',
                'source_id': ''
            }
            key = (disease_vi, symptom, 'has_symptom')
            if key not in seen:
                kb.append(entry)
                seen.add(key)

# Save KB to CSV and JSON
csv_path = OUTPUT_DIR / 'knowledge_base.csv'
json_path = OUTPUT_DIR / 'knowledge_base.json'

fieldnames = ['disease','disease_en','entity','entity_en','entity_type','relation','source_type','source_name','source_id']

with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in kb:
        writer.writerow({k: row.get(k,'') for k in fieldnames})

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(kb, f, ensure_ascii=False, indent=2)

print(f'Wrote {len(kb)} KB records to {csv_path} and {json_path}')
