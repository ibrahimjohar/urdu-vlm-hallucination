import json
import random
from pathlib import Path

ANNOTATIONS_PATH = Path("data/annotations/instances_val2014.json")
OUTPUT_PATH = Path("data/processed/urdu_visual_qa_ft.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
TARGET_SAMPLES   = 1500
random.seed(RANDOM_SEED)

#load hall bench image ids to avoid overlap
import pandas as pd
HALLBENCH_PATH = Path("data/processed/urdu_hall_bench_trilingual.csv")
hall_bench_df = pd.read_csv(HALLBENCH_PATH)
hallbench_ids = set(hall_bench_df['image_id'].unique())
print(f"excluding {len(hallbench_ids)} hallbench images from fine-tuning data")

#roman urdu category mapping
ROMAN_URDU_CATEGORIES = {
    "person": "insaan", "bicycle": "cycle", "car": "gaari",
    "motorcycle": "motorcycle", "airplane": "hawai jahaaz", "bus": "bus",
    "train": "train", "truck": "truck", "boat": "kashti",
    "traffic light": "traffic signal", "fire hydrant": "fire hydrant",
    "stop sign": "stop sign", "parking meter": "parking meter",
    "bench": "bench", "bird": "chirya", "cat": "billi", "dog": "kutta",
    "horse": "ghora", "sheep": "bhed", "cow": "gaay", "elephant": "haathi",
    "bear": "bhalu", "zebra": "zebra", "giraffe": "zaraafa",
    "backpack": "bag", "umbrella": "chhatri", "handbag": "handbag",
    "tie": "tie", "suitcase": "suitcase", "frisbee": "frisbee",
    "skis": "skis", "snowboard": "snowboard",
    "sports ball": "khel ka gend", "kite": "patang",
    "baseball bat": "baseball ka balla",
    "baseball glove": "baseball ka dastaana", "skateboard": "skateboard",
    "surfboard": "surfboard", "tennis racket": "tennis ka racket",
    "bottle": "botal", "wine glass": "gilaas", "cup": "cup",
    "fork": "kaanta", "knife": "churi", "spoon": "chamach", "bowl": "bowl",
    "banana": "kela", "apple": "seb", "sandwich": "sandwich",
    "orange": "santara", "broccoli": "broccoli", "carrot": "gajar",
    "hot dog": "hot dog", "pizza": "pizza", "donut": "donut",
    "cake": "cake", "chair": "kursi", "couch": "sofa",
    "potted plant": "gamla", "bed": "bistar",
    "dining table": "khane ki mez", "toilet": "toilet", "tv": "tv",
    "laptop": "laptop", "mouse": "mouse", "remote": "remote",
    "keyboard": "keyboard", "cell phone": "mobile", "microwave": "microwave",
    "oven": "oven", "toaster": "toaster", "sink": "sink",
    "refrigerator": "fridge", "book": "kitaab", "clock": "ghadi",
    "vase": "phool daani", "scissors": "qainchi", "teddy bear": "teddy bear",
    "hair drier": "hair drier", "toothbrush": "toothbrush"
}

#urdu category mapping (script)
URDU_CATEGORIES = {
    "person": "شخص", "bicycle": "سائیکل", "car": "کار",
    "motorcycle": "موٹر سائیکل", "airplane": "ہوائی جہاز", "bus": "بس",
    "train": "ٹرین", "truck": "ٹرک", "boat": "کشتی",
    "traffic light": "ٹریفک سگنل", "fire hydrant": "فائر ہائیڈرنٹ",
    "stop sign": "اسٹاپ سائن", "parking meter": "پارکنگ میٹر",
    "bench": "بینچ", "bird": "چڑیا", "cat": "بلی", "dog": "کتا",
    "horse": "گھوڑا", "sheep": "بھیڑ", "cow": "گائے",
    "elephant": "ہاتھی", "bear": "ریچھ", "zebra": "زیبرا",
    "giraffe": "زرافہ", "backpack": "بیگ", "umbrella": "چھتری",
    "handbag": "ہینڈ بیگ", "tie": "ٹائی", "suitcase": "سوٹ کیس",
    "frisbee": "فریسبی", "skis": "اسکیز", "snowboard": "اسنوبورڈ",
    "sports ball": "کھیل کی گیند", "kite": "پتنگ",
    "baseball bat": "بیس بال کا بلا",
    "baseball glove": "بیس بال کا دستانہ", "skateboard": "اسکیٹ بورڈ",
    "surfboard": "سرف بورڈ", "tennis racket": "ٹینس کا ریکٹ",
    "bottle": "بوتل", "wine glass": "گلاس", "cup": "کپ",
    "fork": "کانٹا", "knife": "چھری", "spoon": "چمچ", "bowl": "پیالہ",
    "banana": "کیلا", "apple": "سیب", "sandwich": "سینڈوچ",
    "orange": "سنترا", "broccoli": "بروکلی", "carrot": "گاجر",
    "hot dog": "ہاٹ ڈاگ", "pizza": "پیزا", "donut": "ڈونٹ",
    "cake": "کیک", "chair": "کرسی", "couch": "صوفہ",
    "potted plant": "گملا", "bed": "بستر", "dining table": "کھانے کی میز",
    "toilet": "ٹوائلٹ", "tv": "ٹی وی", "laptop": "لیپ ٹاپ",
    "mouse": "ماؤس", "remote": "ریموٹ", "keyboard": "کی بورڈ",
    "cell phone": "موبائل", "microwave": "مائیکروویو", "oven": "اوون",
    "toaster": "ٹوسٹر", "sink": "سنک", "refrigerator": "فریج",
    "book": "کتاب", "clock": "گھڑی", "vase": "پھول دانی",
    "scissors": "قینچی", "teddy bear": "ٹیڈی بیئر",
    "hair drier": "ہیئر ڈرائر", "toothbrush": "ٹوتھ برش"
}

#question-answer templates

def make_identification_sample(image_id, file_name, objects):
    #what objects are in this image
    obj_list_ur = "، ".join([URDU_CATEGORIES.get(o, o) for o in objects])
    obj_list_roman = "، ".join([ROMAN_URDU_CATEGORIES.get(o, o) for o in objects])
    return {
        "id": f"ft_{image_id}_identification",
        "image": file_name,
        "conversations_ur": [
            {"from": "human", "value": "<image>\nاس تصویر میں کیا کیا چیزیں ہیں؟"},
            {"from": "assistant",   "value": f"اس تصویر میں {obj_list_ur} ہیں۔"}
        ],
        "conversations_roman": [
            {"from": "human", "value": "<image>\nIs tasveer mein kya kya cheezein hain?"},
            {"from": "assistant",   "value": f"Is tasveer mein {obj_list_roman} hain."}
        ]
    }

def make_counting_sample(image_id, file_name, objects, counts):
    #count a specific object
    obj = random.choice(list(counts.keys()))
    count = counts[obj]
    obj_ur = URDU_CATEGORIES.get(obj, obj)
    obj_roman = ROMAN_URDU_CATEGORIES.get(obj, obj)
    return {
        "id": f"ft_{image_id}_counting",
        "image": file_name,
        "conversations_ur": [
            {"from": "human", "value": f"<image>\nاس تصویر میں کتنے {obj_ur} ہیں؟"},
            {"from": "assistant",   "value": f"اس تصویر میں {count} {obj_ur} ہیں۔"}
        ],
        "conversations_roman": [
            {"from": "human", "value": f"<image>\nIs tasveer mein kitne {obj_roman} hain?"},
            {"from": "assistant",   "value": f"Is tasveer mein {count} {obj_roman} hain."}
        ]
    }

def make_presence_sample(image_id, file_name, objects, all_categories):
    #yes/no presence — pick a present object
    obj = random.choice(list(objects))
    obj_ur = URDU_CATEGORIES.get(obj, obj)
    obj_roman = ROMAN_URDU_CATEGORIES.get(obj, obj)
    return {
        "id": f"ft_{image_id}_presence",
        "image": file_name,
        "conversations_ur": [
            {"from": "human", "value": f"<image>\nکیا اس تصویر میں {obj_ur} ہے؟"},
            {"from": "assistant",   "value": f"جی ہاں، اس تصویر میں {obj_ur} ہے۔"}
        ],
        "conversations_roman": [
            {"from": "human", "value": f"<image>\nKya is tasveer mein {obj_roman} hai?"},
            {"from": "assistant",   "value": f"Ji haan, is tasveer mein {obj_roman} hai."}
        ]
    }

def make_description_sample(image_id, file_name, objects):
    #general scene description
    obj_list_ur = "، ".join([URDU_CATEGORIES.get(o, o) for o in list(objects)[:4]])
    obj_list_roman = "، ".join([ROMAN_URDU_CATEGORIES.get(o, o) for o in list(objects)[:4]])
    return {
        "id": f"ft_{image_id}_description",
        "image": file_name,
        "conversations_ur": [
            {"from": "human", "value": "<image>\nاس تصویر میں کیا ہو رہا ہے؟"},
            {"from": "assistant",   "value": f"اس تصویر میں {obj_list_ur} نظر آ رہے ہیں۔"}
        ],
        "conversations_roman": [
            {"from": "human", "value": "<image>\nIs tasveer mein kya ho raha hai?"},
            {"from": "assistant",   "value": f"Is tasveer mein {obj_list_roman} nazar aa rahe hain."}
        ]
    }

#load coco annotations
print("loading coco annotations...")
with open(ANNOTATIONS_PATH) as f:
    coco = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco['categories']}
image_filenames = {img['id']: img['file_name'] for img in coco['images']}

#build image_id -> objects and counts
print("building image-object mapping...")
image_objects = {}
image_counts  = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    cat_name = categories[ann['category_id']]
    image_objects.setdefault(img_id, set()).add(cat_name)
    image_counts.setdefault(img_id, {})
    image_counts[img_id][cat_name] = image_counts[img_id].get(cat_name, 0) + 1

#filter out hallbench images and require at least 3 objects
valid_images = [
    (img_id, objs)
    for img_id, objs in image_objects.items()
    if img_id not in hallbench_ids and len(objs) >= 3
]
print(f"valid training images available: {len(valid_images)}")

#generate samples
print(f"generating {TARGET_SAMPLES} fine-tuning samples...")
all_categories = list(categories.values())
samples = []

#shuffle for randomness
random.shuffle(valid_images)

for img_id, objects in valid_images:
    if len(samples) >= TARGET_SAMPLES:
        break

    file_name = image_filenames[img_id]
    counts = image_counts[img_id]

    #generate one of each type per image, pick randomly
    sample_type = random.choice(['identification', 'counting', 'presence', 'description'])

    if sample_type == 'identification':
        samples.append(make_identification_sample(img_id, file_name, list(objects)[:5]))
    elif sample_type == 'counting':
        samples.append(make_counting_sample(img_id, file_name, objects, counts))
    elif sample_type == 'presence':
        samples.append(make_presence_sample(img_id, file_name, objects, all_categories))
    elif sample_type == 'description':
        samples.append(make_description_sample(img_id, file_name, objects))

#save
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\ndone.")
print(f"total samples generated : {len(samples)}")
print(f"saved to                : {OUTPUT_PATH}")

#quick sample check
print("\nsample entries:")
for s in samples[:3]:
    print(f"\n  id    : {s['id']}")
    print(f"  image : {s['image']}")
    print(f"  UR Q  : {s['conversations_ur'][0]['value']}")
    print(f"  UR A  : {s['conversations_ur'][1]['value']}")
    print(f"  ROM Q : {s['conversations_roman'][0]['value']}")
    print(f"  ROM A : {s['conversations_roman'][1]['value']}")