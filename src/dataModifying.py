import pandas as pd
import json
import re
import ast
import tokenize
import io
from black import format_str, FileMode
import requests
import datetime


def remove_comments_from_code(source_code):
    if pd.isna(source_code):
        return source_code

    # Remove whole-line comments
    code = re.sub(r'^\s*#.*$', '', source_code, flags=re.MULTILINE)
    # Remove inline comments (if they follow code)
    code = re.sub(r'\s*#.*', '', code)

    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        # Keep the line if it's not empty OR it's not a direct duplicate of the previous one
        if line.strip() or (cleaned_lines and cleaned_lines[-1] != ''):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def rename_code(source_code):
    if pd.isna(source_code):
        return None

    try:
        source_code = source_code.replace('\r\n', '\n').replace('\r', '\n')
        tree = ast.parse(source_code)
        identifiers = []

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and len(node.id) > 1 and not hasattr(__builtins__,
                                                                                                             node.id):
                    identifiers.append(node.id)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                if len(node.name) > 1 and not hasattr(__builtins__, node.name):
                    identifiers.append(node.name)
                self.generic_visit(node)

        NameCollector().visit(tree)

        unique_identifiers = sorted(set(identifiers))
        replacement_map = {name: chr(97 + i) for i, name in enumerate(unique_identifiers)}

        new_tokens = []
        tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)

        for tok in tokens:
            if tok.type == tokenize.NAME and tok.string in replacement_map:
                new_token = tokenize.TokenInfo(tok.type, replacement_map[tok.string], tok.start, tok.end, tok.line)
            else:
                new_token = tokenize.TokenInfo(tok.type, tok.string, tok.start, tok.end, tok.line)
            new_tokens.append(new_token)

        # Decode from bytes to string
        modified_code = tokenize.untokenize(new_tokens).decode('utf-8')
        return modified_code
    except SyntaxError:
        return None

def reformat_code(source_code):
    try:
        formatted_code = format_str(source_code, mode=FileMode())
        return formatted_code.strip()
    except Exception:
        return None


def create_labeled_dataset(file_path, output_path, max_entries):
    df = pd.read_csv(file_path)
    if "code" not in df.columns or "ai_generated_code" not in df.columns:
        raise ValueError("CSV must contain 'code' and 'ai_generated_code' columns.")

    human_code = df['code'].dropna().astype(str).head(max_entries).tolist()
    ai_code = df['ai_generated_code'].dropna().astype(str).head(max_entries).tolist()

    data = [{"text": text, "label": 0} for text in human_code] + \
           [{"text": text, "label": 1} for text in ai_code]

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"JSONL file saved at: {output_path}")


def remove_comments(file_path, output_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(remove_comments_from_code)
    df['ai_generated_code'] = df['ai_generated_code'].apply(remove_comments_from_code)
    df = df.dropna(subset=['code', 'ai_generated_code'])
    df.to_csv(output_path, index=False)

def check_comments(source_code):
    comment_amount = len(re.findall(r'^\s*#.*$', source_code, flags=re.MULTILINE))
    comment_amount += len(re.findall(r'^(?!\s*#).*?\s#.*', source_code, flags=re.MULTILINE))
    return comment_amount
    
def rename_identifiers(file_path, output_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(rename_code)
    df['ai_generated_code'] = df['ai_generated_code'].apply(rename_code)
    df = df.dropna(subset=['code', 'ai_generated_code'])
    df.to_csv(output_path, index=False)

def reformat(file_path, output_path):
    df = pd.read_csv(file_path)
    df['code'] = df['code'].apply(reformat_code)
    df['ai_generated_code'] = df['ai_generated_code'].apply(reformat_code)
    df = df.dropna(subset=['ai_generated_code'])
    df.to_csv(output_path, index=False)

def keep_correct(file_path, output_path):
    df = pd.read_csv(file_path)
    df = df[df['verdict'] == 'OK']
    df.to_csv(output_path, index=False)

def remove_new_contests(file_path, output_path):
    # Load your data
    df = pd.read_csv(file_path)

    # Fetch all contests from Codeforces
    url = "https://codeforces.com/api/contest.list"
    response = requests.get(url)
    data = response.json()

    if data['status'] != 'OK':
        print("Failed to fetch contest list.")
        return

    contest_dates = {}
    for contest in data['result']:
        if 'startTimeSeconds' in contest:
            contest_id = contest['id']
            start_date = datetime.datetime.utcfromtimestamp(contest['startTimeSeconds']).date()
            contest_dates[contest_id] = start_date

    # Filter DataFrame based on contest date
    def is_old_contest(contest_id):
        date = contest_dates.get(contest_id)
        if date is not None and date > datetime.date(2020, 12, 31):
            print(date)
        return date is not None and date <= datetime.date(2020, 12, 31)

    filtered_df = df[df['contestId'].apply(is_old_contest)]

    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")


def print_data(file_path):
    df = pd.read_csv(file_path)
    for row in df['ai_generated_code'][:20]:
        print("\n", row)
        print("--------------------------------------------")


def compare_datasets(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Merge on index to compare corresponding rows
    merged = pd.merge(df1[['code']], df2[['code']],
                      left_index=True, right_index=True, suffixes=('_df1', '_df2'))
    diff = merged[merged['code_df1'] != merged['code_df2']]
    diff.reset_index(inplace=True)

    for i, row in diff.head(20).iterrows():
        print("-" * 200)
        code1 = row['code_df1'].splitlines()
        code2 = row['code_df2'].splitlines()

        # Print side by side line by line
        max_len = max(len(code1), len(code2))
        for j in range(max_len):
            left = code1[j] if j < len(code1) else ""
            right = code2[j] if j < len(code2) else ""
            print(left.ljust(100) + right)

def clear_dataset(file_path):
    df = pd.read_csv(file_path)
    df['ai_generated_code'] = None
    df.to_csv(file_path, index=False)


def testfunction(file_path):
    df = pd.read_csv(file_path)

    min, index, x_smol = 999, 0, None
    for i, x in enumerate(df['ai_generated_code']):
        lines = len(x.splitlines())
        if lines == 10 and "def" in x:
            print("sample: ", i, "code:\n", x, "\n\n")
        if 9 < lines < min:
            min = lines
            x_smol = x
            index = i
    print(index, min, x_smol)
    print("------------")
    print(df['ai_generated_code'][index])


def remove_empty(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['code', 'ai_generated_code'])
    df.to_csv(output_path, index=False)

def main():
    dataset_name = "baseline"
    remove_comments(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}_no_comments.csv")
    rename_identifiers(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}_replace_identifiers.csv")
    reformat(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}_reformatted.csv")
    keep_correct(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}_correct.csv")

    remove_new_contests(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}.csv")
    create_labeled_dataset(f"../datasets/{dataset_name}.csv", f"../training/labeled_dataset_{dataset_name}.jsonl", 6000)
    print_data(f"../datasets/{dataset_name}.csv")
    compare_datasets("../datasets/baseline.csv", "../datasets/formatted.csv")
    print_data(f"../datasets/{dataset_name}.csv")

    # clear_dataset(f"../datasets/{dataset_name}.csv")
    # remove_empty(f"../datasets/{dataset_name}.csv", f"../datasets/{dataset_name}.csv")


if __name__ == "__main__":
    main()
