import os
import pandas as pd
import json

# Root directory where your year-labeled folders are located
# Use a raw string or forward slashes to avoid Windows path escape issues
INPUT_ROOT = r"C:\Users\Zoro\Desktop\McD_RAG\data"   # adjust if your folder structure is different
OUTPUT_FILE = r"C:\Users\Zoro\Desktop\McD_RAG\data\chunks.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def find_all_csv_files(root_dir):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirpath, fname))
    return csv_files

def serialize_group(category, items):
    items_text = "; ".join(items)
    return f"Category: {category}\nItems: {items_text}"

def main():
    all_chunks = []
    all_csv_files = find_all_csv_files(INPUT_ROOT)
    print(f"Found {len(all_csv_files)} CSV files to process.")

    for input_path in all_csv_files:
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Could not read {input_path}: {e}")
            continue
        
        print(f"Processing {input_path} with columns: {df.columns.tolist()}")

        # Check your actual columns
        required_columns = {"Menu Item", "Category", "Price"}

        if not required_columns.issubset(df.columns):
            print(f"Skipping {input_path}: required columns missing.")
            continue

        # Group only by Category, since we don't have Menu
        grouped = df.groupby("Category")

        for category, group_df in grouped:
            items = group_df["Menu Item"].astype(str).tolist()
            chunk_text = serialize_group(category, items)
            metadata = {
                "category": category,
                "source_file": os.path.relpath(input_path, INPUT_ROOT)
            }
            all_chunks.append({"text": chunk_text, "metadata": metadata})

    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
