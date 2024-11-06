import ast
import astor
import re
import tokenize

from io import StringIO


# def remove_comments(input_string):
#     pass


# def tokenize_code(input_str: str) -> str:
#     """
#     Tokenizes a Python function into a format with <TAB> markers for indentation and spaces
#     between terms.
#
#     Args:
#         input_str (str): The Python code as a single string.
#
#     Returns:
#         str: Tokenized code with <TAB> markers and spaces between terms.
#     """
#     result = []
#     indentation_level = 0
#
#     # Use StringIO to convert the string to a readable stream for tokenize
#     tokens = tokenize.generate_tokens(StringIO(input_str).readline)
#
#     for token in tokens:
#         tok_type = token.type
#         tok_string = token.string
#
#         if tok_type == tokenize.INDENT:
#             indentation_level += 1
#         elif tok_type == tokenize.DEDENT:
#             indentation_level -= 1
#         elif tok_type == tokenize.NEWLINE or tok_type == tokenize.NL:
#             # Add newline with appropriate indentation level
#             result.append("<TAB> " * indentation_level)
#         elif tok_type == tokenize.COMMENT:
#             continue  # Skip comments
#         else:
#             # For operators and punctuation, ensure spaces around symbols
#             if tok_string in {',', ':', '(', ')', '[', ']', '{', '}', '=', '+', '-', '*', '/', '%', '.', '==', '!=',
#                               '<', '>', '<=', '>='}:
#                 result.append(f" {tok_string} ")
#             else:
#                 result.append(tok_string)
#
#     # Join and reformat to remove unnecessary spaces
#     tokenized_code = " ".join(result).replace("  ", " ").strip()
#
#     # Convert multiple consecutive <TAB> lines into a single <TAB> per indentation level
#     # tokenized_code = tokenized_code.replace(" <TAB> ", "<TAB>").replace("<TAB><TAB>", "<TAB>")
#
#     return tokenized_code


def count_token(tokenized_code: str) -> int:
    return len(tokenized_code.split())


# def remove_comment(input_string: str) -> str:
#     """
#     Removes all comment blocks and inline comments from a Python function string.
#
#     Args:
#         input_string (str): The Python function as a string with comments and __NEW_LINE__ as newline markers.
#
#     Returns:
#         str: The function string with comments removed, with __NEW_LINE__ as line breaks.
#     """
#     # Replace __NEW_LINE__ with actual newlines for processing
#     input_string = input_string.replace('__NEW_LINE__', '\n')
#
#     # Remove multi-line comments ("""...""" or '''...''')
#     input_string = re.sub(r'"""(.*?)"""', '', input_string, flags=re.DOTALL)
#     input_string = re.sub(r"'''(.*?)'''", '', input_string, flags=re.DOTALL)
#
#     # Remove single-line comments starting with #
#     input_string = re.sub(r'#.*', '', input_string)
#
#     # Remove any excess whitespace or empty lines resulting from comment removal
#     input_string = re.sub(r'\n\s*\n', '\n', input_string)
#
#     # Replace newlines back to __NEW_LINE__
#     return input_string.replace('\n', '__NEW_LINE__')



# import re
#
#
# def replace_double_quotation_by_single_and_remove_extra_line_split(input_string: str) -> str:
#     """
#     Replaces pairs of double quotation marks with single ones if there are no unescaped single quotes inside
#     and removes line splits for long lines in Python code.
#
#     Args:
#         input_string (str): Original Python code string.
#
#     Returns:
#         str: Modified string with single quotes and joined lines for long statements.
#     """
#     # Replace pairs of double quotes with single quotes if there are no unescaped single quotes inside
#     def replace_quotes(match):
#         content = match.group(0)
#         # Only replace if there are no unescaped single quotes
#         if not re.search(r'(?<!\\)"', content):
#             return content.replace("'", '"')
#         return content
#
#     # Find all double-quoted substrings and apply the replacement function
#     input_string = re.sub(r"'(.*?)'", replace_quotes, input_string)
#
#     # Join lines that are split with an indentation or continuation indicator
#     lines = input_string.splitlines()
#     joined_lines = []
#     for i in range(len(lines)):
#         line = lines[i]
#         # Check for line continuation by indentation or parenthesis continuation
#         if line.strip() and (line.endswith("\\") or line.lstrip() != line):
#             if joined_lines:
#                 joined_lines[-1] += " " + line.strip("\\").strip()
#             else:
#                 joined_lines.append(line.strip("\\").strip())
#         else:
#             joined_lines.append(line)
#
#     # Re-assemble lines, preserving individual line breaks except for those joined
#     return "\n".join(joined_lines)
#
# def remove_comment(input_string: str) -> str:
#     """
#     Removes all comment blocks and inline comments from a Python function string.
#
#     Args:
#         input_string (str): The Python function as a string with comments and __NEW_LINE__ as newline markers.
#
#     Returns:
#         str: The function string with comments removed, with __NEW_LINE__ as line breaks.
#     """
#     # Replace __NEW_LINE__ with actual newlines for processing
#     input_string = input_string.replace('__NEW_LINE__', '\n')
#
#     # Remove multi-line comments ("""...""" or '''...''')
#     input_string = re.sub(r'"""(.*?)"""', '', input_string, flags=re.DOTALL)
#     input_string = re.sub(r"'''(.*?)'''", '', input_string, flags=re.DOTALL)
#
#     # Remove single-line comments starting with #
#     input_string = re.sub(r'#.*', '', input_string)
#
#     # Remove any excess whitespace or empty lines resulting from comment removal
#     input_string = re.sub(r'\n\s*\n', '\n', input_string)
#
#     # Replace newlines back to __NEW_LINE__
#     return input_string.replace('\n', '__NEW_LINE__')
#
#
# def remove_prefix_spaces(func_string: str):
#     parts = func_string.split("\n")
#     parts = [item for idx, item in enumerate(parts) if item != ""]
#     parts = parts + [""] if parts[-1] != "" else parts
#     prefix_n = len(parts[0]) - len(parts[0].lstrip(" "))
#
#     def remove_line(one_line, n):
#         if len(one_line) < n:
#             return one_line
#         elif one_line[:n] == " " * n:
#             return one_line[n:]
#         else:
#             return one_line
#     parts = [remove_line(item, prefix_n) for item in parts]
#     return "\n".join(parts)
#
#
#
if __name__ == "__main__":
    func_string = """def init(self):
    self.__item = None
    self.__baseItem = None
    self.__charge = None
    self.__mutaplasmid = None
    self.__slot = self.dummySlot
    if self.itemID:
        self.__item = eos.db.getItem(self.itemID)
        if self.__item is None:
            pyfalog.error('Item (id: {0}) does not exist', self.itemID)
            return
    if self.baseItemID:
        self.__item = eos.db.getItemWithBaseItemAttribute(self.itemID, self
            .baseItemID)
        self.__baseItem = eos.db.getItem(self.baseItemID)
        self.__mutaplasmid = eos.db.getMutaplasmid(self.mutaplasmidID)
        if self.__baseItem is None:
            pyfalog.error('Base Item (id: {0}) does not exist', self.itemID)
            return
    if self.isInvalid:
        pyfalog.error('Item (id: {0}) is not a Module', self.itemID)
        return
    if self.chargeID:
        self.__charge = eos.db.getItem(self.chargeID)
    self.build()
    """
    print(tokenize_code(func_string))