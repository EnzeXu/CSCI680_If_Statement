from difflib import SequenceMatcher


def sequence_matcher(str1, str2):
    # Compute similarity ratio
    matcher = SequenceMatcher(None, str1, str2)
    similarity = matcher.ratio()
    return similarity
