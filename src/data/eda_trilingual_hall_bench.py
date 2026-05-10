import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/processed/urdu_hall_bench_trilingual.csv")
df = pd.read_csv(CSV_PATH)

print("URDU HALL BENCH - EDA REPORT")

#basic shape
print(f"\n[1] BASIC INFO")
print(f"Total rows : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns : {list(df.columns)}")

#missing values
print(f"\n[2] MISSING VALUES")
missing = df.isnull().sum()
print(missing)
print(f"Any missing : {missing.any()}")

#unique counts
print(f"\n[3] UNIQUE COUNTS")
print(f"Unique images : {df['image_id'].nunique()}")
print(f"Unique categories : {df['category'].nunique()}")
print(f"Unique questions : {df['question_en'].nunique()}")

#split breakdown
print(f"\n[4] SPLIT BREAKDOWN")
print(df.groupby('split').size())

#balance per split
print(f"\n[5] ANSWER BALANCE PER SPLIT")
print(df.groupby(['split', 'answer']).size().unstack(fill_value=0))

#balance overall
print(f"\n[6] OVERALL ANSWER BALANCE")
print(df['answer'].value_counts())
yes_pct = (df['answer'] == 'yes').mean() * 100
no_pct = (df['answer'] == 'no').mean()  * 100
print(f"Yes % : {yes_pct:.1f}%")
print(f"No % : {no_pct:.1f}%")

#questions per image
print(f"\n[7] QUESTIONS PER IMAGE")
qpi = df.groupby('image_id').size()
print(f"Min questions/image : {qpi.min()}")
print(f"Max questions/image : {qpi.max()}")
print(f"Mean questions/img : {qpi.mean():.1f}")

#duplicate questions
print(f"\n[8] DUPLICATE CHECK")
dupes = df.duplicated(subset=['image_id', 'question_en']).sum()
print(f"Duplicate rows : {dupes}")

#sample rows
print(f"\n[9] SAMPLE ROWS (5 random)")
print(df.sample(5).to_string(index=False))

print(f"\n[10] DUPLICATES WITHIN EACH SPLIT")
for split in df['split'].unique():
    split_df = df[df['split'] == split]
    dupes = split_df.duplicated(subset=['image_id', 'question_en']).sum()
    print(f"{split:15s} : {dupes} duplicates")

print("\n\nEDA COMPLETE")