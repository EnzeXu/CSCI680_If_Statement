import os
import re
import random
import tokenize

import pandas as pd
from tqdm import tqdm
from io import StringIO

def remove_comment(input_string: str) -> str:
    """
    Removes all comment blocks and inline comments from a Python function string.

    Args:
        input_string (str): The Python function as a string with comments and __NEW_LINE__ as newline markers.

    Returns:
        str: The function string with comments removed, with __NEW_LINE__ as line breaks.
    """
    # Replace __NEW_LINE__ with actual newlines for processing
    # input_string = input_string.replace('__NEW_LINE__', '\n')

    # Remove multi-line comments ("""...""" or '''...''')
    input_string = re.sub(r'"""(.*?)"""', '', input_string, flags=re.DOTALL)
    input_string = re.sub(r"'''(.*?)'''", '', input_string, flags=re.DOTALL)

    # Remove single-line comments starting with #
    input_string = re.sub(r'#.*', '', input_string)

    # Remove any excess whitespace or empty lines resulting from comment removal
    input_string = re.sub(r'\n\s*\n', '\n', input_string)

    # Replace newlines back to __NEW_LINE__
    # input_string = input_string.replace('\n', '__NEW_LINE__')
    return input_string

def contains_non_english_chars(input_str):
    # Check for any characters outside the ASCII range
    return bool(re.search(r'[^\x00-\x7F]', input_str))


def tokenize_code(input_str: str) -> str:
    """
    Tokenizes a Python function into a format with <TAB> markers for indentation and spaces
    between terms.

    Args:
        input_str (str): The Python code as a single string.

    Returns:
        str: Tokenized code with <TAB> markers and spaces between terms.
    """
    result = []
    indentation_level = 0

    # Use StringIO to convert the string to a readable stream for tokenize
    placeholder = "__PLACEHOLDER_FILL_IN__"
    input_str = input_str.replace("<fill-in>", placeholder)
    tokens = tokenize.generate_tokens(StringIO(input_str).readline)

    for token in tokens:
        tok_type = token.type
        # print(token, tok_type)
        tok_string = token.string

        if tok_type == tokenize.INDENT:
            # Increase indentation level and add <TAB> markers
            indentation_level += 1
            result.append("<TAB> " * indentation_level)
        elif tok_type == tokenize.DEDENT:
            # Decrease indentation level
            indentation_level -= 1
        elif tok_type == tokenize.NEWLINE or tok_type == tokenize.NL:
            # Add newline with appropriate indentation level for next line
            result.append("<TAB> " * indentation_level)
        elif tok_type == tokenize.COMMENT:
            continue  # Skip comments
        else:
            # For operators and punctuation, ensure spaces around symbols
            if tok_string in {',', ':', '(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/', '%', '.', '==', '!=',
                              '<', '>', '<=', '>='}:
                result.append(f" {tok_string} ")
            else:
                result.append(tok_string)

    # Join and reformat to remove unnecessary spaces
    tokenized_code = " ".join(result).replace("  ", " ").strip()
    tokenized_code = tokenized_code.replace(placeholder, "<fill-in>")

    # Convert multiple consecutive <TAB> lines into a single <TAB> per indentation level
    # tokenized_code = tokenized_code.replace(" <TAB> ", "<TAB>").replace("<TAB><TAB>", "<TAB>")

    return tokenized_code


def process_one_python_function_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    code = "\n".join(lines)
    code = remove_comment(code)
    flatten_code, modified_code, chosen_line = replace_random_if_condition_with_indentation(code)
    if not chosen_line:
        return code, code, None

    # print("=" * 200)
    # print(f"'{modified_source}'")
    # print("#" * 200)
    # print(f"'{tokenize_code(modified_source)}'")
    # print("=" * 200)
    # print(f"'{chosen_line}'")
    # print("#" * 200)
    # print(f"'{tokenize_code(chosen_line)}'")
    # print("=" * 200)
    return tokenize_code(flatten_code), tokenize_code(modified_code), tokenize_code(chosen_line)
    # print(len(code))


def replace_random_if_condition_with_indentation(input_string):
    # Find all matches for if conditions with preceding spaces or tabs
    if contains_non_english_chars(input_string):
        return input_string, input_string, None
    matches = list(re.finditer(r'(\n)([ \t]*)(if[^\n]*)', input_string, re.MULTILINE))

    if not matches:
        return input_string, input_string, None  # No matches found

    # Randomly pick one match
    chosen_match = random.choice(matches)
    newline = chosen_match.group(1)  # The newline character
    indentation = chosen_match.group(2)  # The spaces or tabs matched by [ \t]
    chosen_line = chosen_match.group(0)  # Extract the entire matched line as a string

    # Replace the chosen line with "<fill-in>" and keep the original indentation
    modified_source = (
            input_string[:chosen_match.start()] +
            f"{newline}{indentation}<fill-in>" +
            input_string[chosen_match.end():]
    )
    chosen_line = chosen_line.lstrip()
    input_string = input_string.lstrip()
    return input_string, modified_source, chosen_line


def process_data_token(input_folder, save_path="src/full_dataset.csv"):
    file_list = os.listdir(input_folder)
    file_list = sorted(file_list)
    file_list = file_list
    input_method_all, input_method_if_statement, target_block = [], [], []
    with tqdm(total=len(file_list)) as progress_bar:
        for idx, one_file_name in enumerate(file_list):
            progress_bar.set_description(f"{one_file_name}")

            one_path = os.path.join(input_folder, one_file_name)
            try:
                flatten_code, modified_code, chosen_line = process_one_python_function_file(one_path)
                if chosen_line:
                    input_method_all.append(flatten_code)
                    input_method_if_statement.append(modified_code)
                    target_block.append(chosen_line)
            except Exception as e:
                print(f"[Error in process_data_token] {e}")

            progress_bar.update(1)
    df = pd.DataFrame()
    df["input_method_all"] = input_method_all
    df["input_method_if_statement"] = input_method_if_statement
    df["target_block"] = target_block
    df.reset_index(drop=True)
    print(f"df length: {df}")
    df.to_csv(save_path)


if __name__ == "__main__":
    process_data_token("data/")