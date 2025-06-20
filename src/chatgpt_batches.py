from tqdm import tqdm
import pandas as pd
import json
import os
import re
import math
import numpy as np
from openai import OpenAI
client = OpenAI()

def create_prompt(df, row):
    prompt = (
        f"!Only return the Python code in a Markdown block. Keep the usual amount of comments, but do not add extra explanations outside the code block. Do not remove comments that would normally be included.! Write a python script for the following problem: {df.loc[row, 'problem-description']}\n"
        f"Input specification: {df.loc[row, 'input-specification']}. "
        f"Output specification: {df.loc[row, 'output-specification']}\n"
        f"Demo input: {df.loc[row, 'demo-input']}, "
        f"Demo output: {df.loc[row, 'demo-output']}\n"
        f"Note: {df.loc[row, 'note']}")
    print(prompt)
    return prompt

def create_prompt_complete(df, row):
    # Extract code and truncate after second or third newline
    code_snippet = df.loc[row, 'code']
    split_code = code_snippet.split('\n')
    num_lines = len(split_code)
    num_lines_to_keep = min(max(math.floor(num_lines / 4),1),8)
    truncated_code = '\n'.join(split_code[:num_lines_to_keep])
    prompt = (
        f"!Only return the Python code in a Markdown block. Keep the usual amount of comments, but do not add extra explanations outside the code block. Do not remove comments that would normally be included.! Complete the given code for the following problem: {df.loc[row, 'problem-description']}\n"
        f"Input specification: {df.loc[row, 'input-specification']}. "
        f"Output specification: {df.loc[row, 'output-specification']}\n"
        f"Demo input: {df.loc[row, 'demo-input']}, "
        f"Demo output: {df.loc[row, 'demo-output']}\n"
        f"Note: {df.loc[row, 'note']}\n"
        f"Complete the following code: ```python\n{truncated_code}\n```")
    print(prompt)
    return prompt


def create_requests_file(start, end, output_file, dataset):
    df = pd.read_csv(f"../datasets/{dataset}.csv", dtype={"ai_generated_code": str})

    with open(output_file, "w") as f:
        for i in range(start,end):
            prompt = create_prompt(df, i)
            request_data = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a Python programmer."},
                        {"role": "user", "content": prompt}
                    ],
                    # "max_tokens": 5000,
                    # "temperature": 1.0
                }
            }
            f.write(json.dumps(request_data) + "\n")
    print(f"JSONL file '{output_file}' has been created successfully.")


def create_batch_file(jsonl_file):
    batch_input_file = client.files.create(
        file=open(jsonl_file, "rb"),
        purpose="batch"
    )

    print(batch_input_file)
    return batch_input_file

def create_batch_request(batch_input_file):
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

def check_batch_status(batch_id):
    batch = client.batches.retrieve(batch_id)
    print(batch)

def get_batch_result(batch_id):
    file_response = client.files.content(batch_id)
    print(file_response.text)

def list_batches():
    print(client.batches.list())

def cancel_batch(batch_id):
    client.batches.cancel(batch_id)

def rename_batches():
    for filename in os.listdir("../batch_files"):
        if filename.startswith("batch_68") and filename.endswith(".jsonl"):
            file_path = os.path.join("../batch_files", filename)

            # Read the first line of the file
            with open(file_path, "r", encoding="utf-8") as file:
                first_line = file.readline().strip()

                try:
                    data = json.loads(first_line)
                    custom_id = data.get("custom_id")

                    if custom_id:
                        new_filename = f"batch_output_{custom_id}_{int(custom_id)+999}.jsonl"
                        new_file_path = os.path.join("../batch_files", new_filename)

                        # Rename the file
                        os.rename(file_path, new_file_path)
                        print(f"Renamed {filename} -> {new_filename}")
                except json.JSONDecodeError:
                    print(f"Skipping {filename}: Invalid JSON format")

def extract_python_code(response_text):
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    return "\n".join(code_blocks).strip() if code_blocks else response_text

def process_batch(file_name, dataset):
    try:
        df = pd.read_csv(f"../datasets/{dataset}.csv", dtype={"ai_generated_code": str})
    except FileNotFoundError:
        print("dataset not found" + dataset)
    if "ai_generated_code" not in df.columns:
        df["ai_generated_code"] = ""
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    custom_id = data.get("custom_id")
                    content = data.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get(
                        "content", "")
                    content = extract_python_code(content)
                    #print(custom_id, content)
                    if custom_id is not None and content:
                        custom_id = int(custom_id)  # Ensure custom_id is an integer index
                        if custom_id in df.index:
                            df.at[custom_id, "ai_generated_code"] = content
                        else:
                            print("given index not found in dataset")
                except (json.JSONDecodeError, IndexError, KeyError, ValueError):
                    print(f"Skipping invalid JSON line in {file_name}")
    except FileNotFoundError:
        print("file not found: " + file_name)

    # Save the updated CSV
    df.to_csv(f"../datasets/{dataset}.csv", index=False)
    print(f"Updated {dataset} with AI-generated content from {file_name}")

def get_max_completion_tokens(file_path):
    max_tokens = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            max_tokens += data["response"]["body"]["usage"]["completion_tokens"]>1000
    return max_tokens


def main():
    # Create Batch requests
    dataset = "baseline"
    start,end = 0,30000
    while start < end:
        temp_end = min(start + 1000, end)  # Ensure we don't exceed 'end'
        file_name = f"../batch_files/requests_file_{dataset}_{start}_{temp_end - 1}.jsonl"

        create_requests_file(start, temp_end, file_name, dataset)
        batch_file = create_batch_file(file_name)
        create_batch_request(batch_file)

        start = temp_end


    # rename_batches()
    # check_batch_status("batch_67c9a5da4224819096c3ffd488a48d26")
    # cancel_batch("batch_67cb195b964c81908e5251a7d1aab37e")
    # list_batches()


    # Process Batches
    dataset = "baseline"
    start, end = 0,30000
    for batch_start in range(start, end, 1000):
        batch_end = batch_start + 999
        batch_file = f"../batch_files/batch_output_{dataset}_{batch_start}_{batch_end}.jsonl"
        process_batch(batch_file, dataset)

if __name__=="__main__":
    main()