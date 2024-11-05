import ast
import astor

from .utils import remove_prefix_spaces, remove_comment, replace_double_quotation_by_single_and_remove_extra_line_split


class DataClean:
    def __init__(self, raw_function_string):
        self.raw_function_string = raw_function_string

    @staticmethod
    def to_original_method(input_str):
        output_str = input_str.replace("\n", "__NEW_LINE__")
        return output_str


def data_clean(original_method: str) -> dict:
    """
    Cleans and processes the original method string, returning a dictionary with
    cleaned method, input method, target block, tokens in method, and flattened method.

    Args:
        original_method (str): The original method string.

    Returns:
        dict: Contains the following fields:
            - cleaned_method (str): Properly formatted version of the method.
            - input_method (str): Method with specific markers for tokens and indentation.
            - target_block (str): Target code block from the method.
            - tokens_in_method (int): Token count in the method.
            - flattened_method (str): Flattened version of the method with delimiters replaced.
    """
    # Example cleaning operations to mimic the fields in the CSV

    # cleaned_method = replace_double_quotation_by_single_and_remove_extra_line_split(original_method)
    cleaned_method = original_method.replace("__NEW_LINE__", "\n").replace("__TAB__", "    ")
    parsed_cleaned_method = ast.parse(cleaned_method)
    cleaned_method = astor.to_source(parsed_cleaned_method)
    # cleaned_method = remove_prefix_spaces(cleaned_method)
    input_method = cleaned_method.replace("\n", "").replace("    ", "<TAB>") #remove_comment(original_method)
    input_method = input_method.replace("__NEW_LINE__", "<TAB>").replace("__TAB__", "<TAB>")
    target_block = "if example_condition:"  # Placeholder for actual target block extraction
    tokens_in_method = len(original_method.split())  # Simplified token count
    flattened_method = original_method.replace("\n", "").replace(" ", "").replace("__NEW_LINE__", "")

    return {
        "cleaned_method": cleaned_method,
        "input_method": input_method,
        "target_block": target_block,
        "tokens_in_method": tokens_in_method,
        "flattened_method": flattened_method
    }

