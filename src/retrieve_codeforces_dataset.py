import pandas as pd
from datasets import load_dataset

print("start")
dataset = load_dataset("MatrixStudio/Codeforces-Python-Submissions", split="train")

columns_of_interest = [
    "contestId", "index", "name", "type", "tags",
    "problem-description", "input-specification",
    "output-specification", "demo-input", "demo-output", "note",
    "verdict", "code"
]

filtered_rows = []
for i, row in enumerate(dataset):
    if i >= 1000:  # Limit to 1000 rows
        break
    # Select only the columns of interest
    filtered_rows.append({col: row.get(col, None) for col in columns_of_interest})

df = pd.DataFrame(filtered_rows)
df.to_csv("../datasets/small/coding_tasks_dataset.csv", index=False)