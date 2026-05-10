import json
from collections import Counter
from pathlib import Path

FT_PATH = Path("data/processed/urdu_visual_qa_ft.json")

#load
print("loading fine-tuning dataset...")
with open(FT_PATH, encoding='utf-8') as f:
    data = json.load(f)

#basic info
print("URDU VISUAL QA FT - EDA REPORT")

print(f"\n[1] BASIC INFO")
print(f"total samples: {len(data)}")

#unique images
images = set(s['image'] for s in data)
print(f"unique images: {len(images)}")

#sample type distribution
print(f"\n[2] SAMPLE TYPE DISTRIBUTION")
types = Counter(s['id'].split('_')[-1] for s in data)
for t, count in sorted(types.items()):
    print(f"  {t:20s} : {count}")

#language coverage
print(f"\n[3] LANGUAGE COVERAGE")
has_ur = all('conversations_ur' in s for s in data)
has_roman = all('conversations_roman' in s for s in data)
print(f"all have urdu script: {has_ur}")
print(f"all have roman urdu: {has_roman}")

#image tag check
print(f"\n[4] IMAGE TAG CHECK")
has_tag_ur = all('<image>' in s['conversations_ur'][0]['value'] for s in data)
has_tag_roman = all('<image>' in s['conversations_roman'][0]['value'] for s in data)
print(f"urdu has image tag: {has_tag_ur}")
print(f"roman has image tag : {has_tag_roman}")

#conversation structure
print(f"\n[5] CONVERSATION STRUCTURE")
turns_ur = all(len(s['conversations_ur']) == 2 for s in data)
turns_roman = all(len(s['conversations_roman']) == 2 for s in data)
print(f"all ur have 2 turns: {turns_ur}")
print(f"all roman have 2 turns: {turns_roman}")

#sample entries
print(f"\n[6] SAMPLE ENTRIES (3 random)")
import random
random.seed(42)
for s in random.sample(data, 3):
    print(f"\n  id     : {s['id']}")
    print(f"  image  : {s['image']}")
    print(f"  UR  Q  : {s['conversations_ur'][0]['value'].strip()}")
    print(f"  UR  A  : {s['conversations_ur'][1]['value']}")
    print(f"  ROM Q  : {s['conversations_roman'][0]['value'].strip()}")
    print(f"  ROM A  : {s['conversations_roman'][1]['value']}")

print("\n\nEDA COMPLETE")