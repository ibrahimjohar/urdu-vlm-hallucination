import pandas as pd
import time
from pathlib import Path
from deep_translator import GoogleTranslator

BILINGUAL_PATH = Path("data/processed/urdu_hall_bench_bilingual.csv")

print("loading existing bilingual file...")
df = pd.read_csv(BILINGUAL_PATH)
print(f"total rows : {len(df)}")
print(f"already translated: {df['question_ur'].notna().sum()}")
print(f"still missing : {df['question_ur'].isna().sum()}")

#translator
translator = GoogleTranslator(source='english', target='urdu')

#find missing rows
missing_mask = df['question_ur'].isna()
missing_indices = df[missing_mask].index.tolist()
print(f"\nretrying {len(missing_indices)} failed rows...\n")

failed_indices = []

for i, idx in enumerate(missing_indices):
    try:
        urdu = translator.translate(df.loc[idx, 'question_en'])
        df.loc[idx, 'question_ur'] = urdu

        #progress update every 100 rows
        if (i + 1) % 100 == 0:
            print(f"  retried {i + 1} / {len(missing_indices)}...")
            #checkpoint save every 100 rows
            df.to_csv(BILINGUAL_PATH, index=False)
            print(f"  checkpoint saved.")

        #small delay to avoid rate limiting
        time.sleep(0.15)

    except Exception as e:
        print(f"  failed again at index {idx}: {e}")
        failed_indices.append(idx)

#final save
df.to_csv(BILINGUAL_PATH, index=False)

#report
print(f"\ntranslation complete.")
print(f"now translated : {df['question_ur'].notna().sum()}")
print(f"still missing  : {df['question_ur'].isna().sum()}")
print(f"failed again   : {len(failed_indices)}")

if failed_indices:
    print(f"failed indices: {failed_indices}")

#quick sample check
print("\nsample translations:")
sample = df[['question_en', 'question_ur']].drop_duplicates().head(10)
for _, row in sample.iterrows():
    print(f"EN: {row['question_en']}")
    print(f"UR: {row['question_ur']}")
    print()