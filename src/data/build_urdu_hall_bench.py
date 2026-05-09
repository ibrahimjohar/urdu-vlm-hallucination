import json
import random
import pandas as pd
from pathlib import Path

ANNOTATIONS_PATH = Path("data/annotations/instances_val2014.json")
OUTPUT_PATH = Path("data/processed/urdu_hall_bench_english.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

#config
NUM_IMAGES = 200
QUESTIONS_PER_IMG = 3   #3 positive + 3 negative = 6 questions per image
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

#load COCO annotations
print("loading COCO annotations...")
with open(ANNOTATIONS_PATH) as f:
    coco = json.load(f)

#build category id → name mapping
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
all_category_names = list(categories.values())

#build image_id → set of present object names
print("building image-object mapping...")
image_objects = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    cat_name = categories[ann["category_id"]]
    image_objects.setdefault(img_id, set()).add(cat_name)

#build image_id → file name mapping
image_filenames = {img["id"]: img["file_name"] for img in coco["images"]}

#count category frequencies for popular split
print("computing category frequencies...")
category_freq = {}
for ann in coco["annotations"]:
    cat = categories[ann["category_id"]]
    category_freq[cat] = category_freq.get(cat, 0) + 1

popular_objects = sorted(category_freq, key=category_freq.get, reverse=True)

#select 200 images w/ atleast 3 distinct objects
print("selecting images...")
valid_images = [
    (img_id, objs)
    for img_id, objs in image_objects.items()
    if len(objs) >= 3
]
selected_images = random.sample(valid_images, NUM_IMAGES)
print(f"selected {len(selected_images)} images")

#generate POPE-style questions
def get_negative_samples(present_objects, split, n):
    absent = [c for c in all_category_names if c not in present_objects]
    if split == "random":
        return random.sample(absent, n)
    elif split == "popular":
        return [o for o in popular_objects if o not in present_objects][:n]
    elif split == "adversarial":
        #simplified adversarial: most popular objects not present
        #will be refined with co-occurrence data later
        return [o for o in popular_objects if o not in present_objects][:n]

print("generating questions...")
all_questions = []

for img_id, present_objects in selected_images:
    present_list = list(present_objects)

    for split in ["random", "popular", "adversarial"]:
        #positive questions - object IS in the image
        pos_objects = random.sample(
            present_list, min(QUESTIONS_PER_IMG, len(present_list))
        )
        for obj in pos_objects:
            all_questions.append({
                "image_id": img_id,
                "file_name": image_filenames[img_id],
                "question_en": f"Is there a {obj} in the image?",
                "answer": "yes",
                "split": split,
                "category": obj
            })

        #negative questions - object is NOT in the image
        neg_objects = get_negative_samples(
            present_objects, split, QUESTIONS_PER_IMG
        )
        for obj in neg_objects:
            all_questions.append({
                "image_id": img_id,
                "file_name": image_filenames[img_id],
                "question_en": f"Is there a {obj} in the image?",
                "answer": "no",
                "split": split,
                "category": obj
            })

#saving to csv
df = pd.DataFrame(all_questions)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\ndone.")
print(f"total questions generated: {len(df)}")
print(f"unique images: {df['image_id'].nunique()}")
print(f"\nsplit breakdown:")
print(df.groupby(["split", "answer"]).size().unstack())
print(f"\nsaved to: {OUTPUT_PATH}")