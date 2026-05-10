import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("data/processed/urdu_hall_bench_bilingual.csv")
OUTPUT_PATH = Path("data/processed/urdu_hall_bench_trilingual.csv")

#roman urdu category mapping
ROMAN_URDU_CATEGORIES = {
    "person": "insaan",
    "bicycle": "cycle",
    "car": "gaari",
    "motorcycle": "motorcycle",
    "airplane": "hawai jahaaz",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "kashti",
    "traffic light": "traffic signal",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    "bird": "chirya",
    "cat": "billi",
    "dog": "kutta",
    "horse": "ghora",
    "sheep": "bhed",
    "cow": "gaay",
    "elephant": "haathi",
    "bear": "bhalu",
    "zebra": "zebra",
    "giraffe": "zaraafa",
    "backpack": "bag",
    "umbrella": "chhatri",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "khel ka gend",
    "kite": "patang",
    "baseball bat": "baseball ka balla",
    "baseball glove": "baseball ka dastaana",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis ka racket",
    "bottle": "botal",
    "wine glass": "gilaas",
    "cup": "cup",
    "fork": "kaanta",
    "knife": "churi",
    "spoon": "chamach",
    "bowl": "bowl",
    "banana": "kela",
    "apple": "seb",
    "sandwich": "sandwich",
    "orange": "santara",
    "broccoli": "broccoli",
    "carrot": "gajar",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "chair": "kursi",
    "couch": "sofa",
    "potted plant": "gamla",
    "bed": "bistar",
    "dining table": "khane ki mez",
    "toilet": "toilet",
    "tv": "tv",
    "laptop": "laptop",
    "mouse": "mouse",
    "remote": "remote",
    "keyboard": "keyboard",
    "cell phone": "mobile",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "refrigerator": "fridge",
    "book": "kitaab",
    "clock": "ghadi",
    "vase": "phool daani",
    "scissors": "qainchi",
    "teddy bear": "teddy bear",
    "hair drier": "hair drier",
    "toothbrush": "toothbrush"
}

#load bilingual benchmark
print("loading bilingual benchmark...")
df = pd.read_csv(INPUT_PATH)
print(f"loaded {len(df)} rows")

#generate roman urdu questions
print("generating roman urdu questions...")

def make_roman_question(category):
    roman_object = ROMAN_URDU_CATEGORIES.get(category, category)
    return f"Kya tasveer mein {roman_object} hai?"

df['question_roman'] = df['category'].apply(make_roman_question)

#check for unmapped categories
unmapped = df[~df['category'].isin(ROMAN_URDU_CATEGORIES)]['category'].unique()
if len(unmapped) > 0:
    print(f"warning — unmapped categories: {unmapped}")
else:
    print("all categories mapped successfully")

# save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nsaved to: {OUTPUT_PATH}")

#quick sample check
print("\nsample questions across all three languages:")
sample = df[['category', 'question_en', 'question_ur', 'question_roman']].drop_duplicates(subset='category').head(10)
for _, row in sample.iterrows():
    print(f"  category : {row['category']}")
    print(f"  EN       : {row['question_en']}")
    print(f"  UR       : {row['question_ur']}")
    print(f"  ROMAN    : {row['question_roman']}")
    print()

print(f"total rows     : {len(df)}")
print(f"total columns  : {len(df.columns)}")
print(f"columns        : {list(df.columns)}")