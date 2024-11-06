import os
import shutil
import re
import json
import subprocess
from tqdm import tqdm
import random
import datetime
import pytz
import pickle

# from repo_names import REPO_NAMES

def extract_and_rename_python_files(repo_url, destination_folder, index, index_dic, progress_bar=None):
    tmp_folder = './tmp'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(tmp_folder, repo_name)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    # print(f"Cloning the repository {repo_url} into {tmp_folder}...")
    # subprocess.run(["git", "clone", repo_url, repo_path])
    try:
        subprocess.run(
            ["git", "clone", repo_url, repo_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20
        )
    except subprocess.TimeoutExpired:
        print(f"Error: Cloning the repository {repo_url} took too long (more than 10 seconds).")
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        return index

    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # print(f"python_files: {python_files}")

    for python_file in python_files:
        if " " in python_file:
            continue
        new_filename = f"{index:08d}.txt"
        destination_path = os.path.join(destination_folder, new_filename)
        # shutil.copy(python_file, destination_path)
        try:
            num_of_func = process_python_file(python_file, destination_path)
        except Exception as e:
            print(f"Errors in file \"{python_file}\"", e)
            num_of_func = 0
        if num_of_func > 0:
            # index_dic["class"][str(index)] = num_of_class
            index_dic["func"][str(index)] = num_of_func
            index_dic["source"][str(index)] = python_file[6:]
            if not progress_bar:
                print(f"index: {index:08d} func: {num_of_func:03d} filename: {python_file[6:]}")
            else:
                progress_bar.set_description(f"index: {index:08d} func: {num_of_func:03d} filename: {python_file[6:]}")
                # print(f"index: {index:08d} func: {num_of_func:03d} filename: {python_file[6:]}")
            index += 1
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    return index


def process_python_file(python_file, destination_path):
    # Read the Python file
    with open(python_file, "r") as f:
        content = f.read()

    # Remove single-line comments
    # content = re.sub(r'#.*?(\n|$)', '', content)
    # print("=" * 200)
    # print(content)

    num_of_func = 0

    # Find all functions and extract them
    # functions_content = ""
    # func_matches = re.finditer(r'\bdef\s+\w+\s*\(.*?\):', content)
    func_matches = re.finditer(r'^def\s+\w+\s*\(.*?\):', content, re.MULTILINE)
    # print(func_matches)

    for func_match in func_matches:
        func_start = func_match.start()
        func_end = find_func_end(content, func_start)
        function_code = content[func_start:func_end]
        if len(function_code) > 2000 or len(function_code) < 500:
            continue
        if not has_one_def_keyword(function_code):
            continue
        # print("=" * 200)
        # print(function_code)
        if re.search(r'\n[ \t]*if', function_code, re.MULTILINE):  # if re.match(r'^\n[ \t]*if', function_code):
            num_of_func += 1
            new_destination_path = destination_path.replace(".txt", f"_{num_of_func:04d}.txt")
            # functions_content += function_code + "\n\n"  # Add some spacing between functions

            # Write the extracted functions to the destination file
            with open(new_destination_path, 'w') as f:
                f.write(function_code)

    return num_of_func


def has_one_def_keyword(function_code):
    # Regex pattern to match "def" at the beginning of a line or following whitespace
    # and followed by a space to ensure it's not part of another word.
    pattern = r'(^|\s)def\s'
    # Find all matches of the pattern in the string
    matches = re.findall(pattern, function_code)
    # Return True if there's only one match, otherwise False
    return len(matches) == 1

def find_func_end(content, func_start):
    """
    Finds the end of a function based on indentation level.
    """
    lines = content[func_start:].splitlines(keepends=True)
    base_indentation = None
    in_function = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Determine the base indentation level on the first non-empty line
        current_indentation = len(line) - len(line.lstrip())
        if base_indentation is None:
            base_indentation = current_indentation
            in_function = True
            continue

        # Check if we are outside the function's indentation level
        if in_function and current_indentation <= base_indentation and not stripped.startswith("#"):
            return func_start + sum(len(l) for l in lines[:i])

    return func_start + sum(len(l) for l in lines)  # End of content if function runs to end of file



def build_dataset_files(raw_repo_names, start, end):
    repo_name_list = raw_repo_names  # .split()
    repo_name_list = repo_name_list[start:end]
    index_save_path = "index.json"

    if os.path.exists(index_save_path):
        with open(index_save_path, "r") as f:
            index_dic = json.load(f)
        index = index_dic["index"]
        print(f"Loaded {index_save_path}: index = {index}")
    else:
        index = 0
        index_dic = dict()
        # index_dic["class"] = dict()
        index_dic["func"] = dict()
        index_dic["source"] = dict()
        index_dic["repo_list"] = []

    with tqdm(total=len(repo_name_list)) as progress_bar:
        # for one_repo_name in tqdm(repo_name_list):
        for idx, one_repo_name in enumerate(repo_name_list):
            if one_repo_name in index_dic["repo_list"]:
                print(f"skip repo {one_repo_name}")
                progress_bar.update(1)
                continue
            else:
                repo_url = f"https://github.com/{one_repo_name}"
                index = extract_and_rename_python_files(repo_url, "data/", index, index_dic, progress_bar)
            progress_bar.update(1)
            index_dic["repo_list"].append(one_repo_name)
            if idx % 20 == 0:
                index_dic["index"] = index

                # sum_class = sum([index_dic["class"][str(i)] for i in range(index) if str(i) in index_dic["class"]])
                sum_func = sum([index_dic["func"][str(i)] for i in range(index) if str(i) in index_dic["func"]])

                # index_dic["sum_class"] = sum_class
                index_dic["sum_func"] = sum_func
                with open(index_save_path, "w") as f:
                    json.dump(index_dic, f, indent=4)

                print(f"idx = {idx} Index file saved to {index_save_path}")
    index_dic["index"] = index

    # sum_class = sum([index_dic["class"][str(i)] for i in range(index) if str(i) in index_dic["class"]])
    sum_func = sum([index_dic["func"][str(i)] for i in range(index) if str(i) in index_dic["func"]])

    # index_dic["sum_class"] = sum_class
    index_dic["sum_func"] = sum_func

    print(f"sum_func: {sum_func}")

    with open(index_save_path, "w") as f:
        json.dump(index_dic, f, indent=4)

    print(f"Index file saved to {index_save_path}")


if __name__ == "__main__":
    with open("ifstatement/dataset/repo_names.pkl", "rb") as f:
        repo_names = pickle.load(f)
    build_dataset_files(repo_names, 8500, 30000)
