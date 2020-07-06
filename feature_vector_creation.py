import sys
import csv
import collections
import string
import math
import re
import pandas as pd
import enchant
from string import ascii_lowercase as al, ascii_uppercase as au, digits as dg, punctuation as pt


"""
Helper Functions and Classes
"""


url_characters = al + au + dg + "$-_+!*'()," # Common characters in an URL


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def get_letters_ratio(original_string):
    """
    Get the ratio of letters in a string
    """
    if len(original_string) > 0:
        return len(list(filter(str.isalpha, original_string))) / len(original_string)
    return 0


def get_digits_ratio(original_string):
    """
    Get the ratio of digits in a string
    """
    if len(original_string) > 0:
        return len(list(filter(lambda ch: not ch.isalpha() and ch.isalnum(), original_string))) / len(original_string)
    return 0


def get_symbols_ratio(original_string):
    """
    Get the ratio of symbols in a string
    """
    if len(original_string) > 0:
        return len(list(filter(lambda ch: not ch.isalnum(), original_string))) / len(original_string)
    return 0


def get_longest_number_string(original_string):
    """
    Get the longest string of consecutive numbers in a string
    For example in 'a1b23c456de7f' it would return '456'
    """
    longest_number_string = ''
    regex = r'([0-9]+)'
    matches = re.findall(regex, original_string)
    if matches:
        longest_number_string = max(matches, key=len)
    return longest_number_string


def get_longest_number_string_ratio(original_string):
    """
    Wrapper for get_longest_number_string
    It returns the ratio compared to the total length
    """
    if len(original_string) > 0:
        return len(get_longest_number_string(original_string)) / len(original_string)
    return 0


def get_longest_letters_string(original_string):
    """
    Get the longest string of consecutive letters in a string
    For example in 'a1b23c456de7f' it would return 'de'
    """
    longest_letters_string = ''
    regex = r'([a-zA-Z]+)'
    matches = re.findall(regex, original_string)
    if matches:
        longest_letters_string = max(matches, key=len)
    return longest_letters_string


def get_longest_letters_string_ratio(original_string):
    """
    Wrapper for get_longest_letters_string
    It returns the ratio compared to the total length
    """
    if len(original_string) > 0:
        return len(get_longest_letters_string(original_string)) / len(original_string)
    return 0


def get_all_substrings(original_string):
    """
    Get all the contiguous substrings in a string
    """
    substrings = []
    for i in range(len(original_string)):
        for j in range(i, len(original_string)):
            substrings.append(original_string[i:j+1])
    return substrings


def has_digits_or_punctuation(original_string):
    """
    Check if a string has any digit or symbols
    """
    return any(char.isdigit() or char in pt for char in original_string)


def get_longest_meaningful_word(original_string):
    """
    Get the longest substring that belongs to the English dictionary
    has_digits_or_punctuation is needed because enchant understands digit
    strings and some symbols as valid words
    """
    dictionary = enchant.Dict('en_US')
    substrings = set(get_all_substrings(original_string))
    longest_meaningful_word = ''
    for substring in substrings:
        if (not has_digits_or_punctuation(substring) and
        dictionary.check(substring.lower()) and
        len(substring) > len(longest_meaningful_word)):
            longest_meaningful_word = substring
    return longest_meaningful_word


def get_longest_meaningful_word_ratio(original_string):
    """
    Wrapper for get_longest_meaningful_word
    It returns the ratio compared to the total length
    """
    if len(original_string) > 0:
        return len(get_longest_meaningful_word(original_string)) / len(original_string)
    return 0


# Iterator to calculate entropies.
# ord(c) returns an integer representing the Unicode character.
def range_url(): return (ord(c) for c in url_characters)


def metric_entropy(data, iterator=range_url):
    """
    Returns the metric entropy (Shannon's entropy divided by string length)
    for some data given a set of possible data elements
    Based on: http://pythonfiddle.com/shannon-entropy-calculation/
    """
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy / len(data)


"""
Functions to create feature vectors. Each function creates a different type of vector.
Several functions are created to check the performance of different feature vectors.
"""


def extract_features_with_letter_counting(query, attack):
    """
    Extract the features for a DNS query string counting all the letters in the string
    in proportion with the total length of the query
    The features are:
        - Count of alphanumeric characters (a: 0.375, b: 0.25, c: 0.125...)
        - Number of non-alphanumeric characters (symbols: 0.125)
        - Longest consecutive number in the string (longest_number: 0.25)
    """
    length = len(query)
    if length > 0:
        # Create dictionary with the number of repetitions of the alphanumeric
        # characters in proportion with the length of the query
        features = {x:(query.count(x) / length) for x in al+dg}
    else:
        # Create emtpy dictionary for empty string
        features = {x:0 for x in al+dg}
    # The symbols in proportion with the total length
    features['symbols'] = get_symbols_ratio(query)
    # Feature that measures the longest string of numbers that are together in proportion with the total length
    features['longest_number'] = get_longest_number_string_ratio(query)
    features['attack'] = attack
    return features


def extract_features_with_letters_and_numbers(query, attack):
    """
    Extract the features for a DNS query string counting all the letters,
    numbers and symbols in proportion with the total length of the query
    The features are:
        - Count of letters (letters: 0.8)
        - Count of numbers (numbers: 0.1)
        - Number of non-alphanumeric characters (symbols: 0.1)
        - Longest consecutive number in the string (longest_number: 0.1)
    """
    features = {}
    # Count the letters
    features['letters'] = get_letters_ratio(query)
    # Count the numbers
    features['numbers'] = get_digits_ratio(query)
    # Count the symbols
    features['symbols'] = get_symbols_ratio(query)
    # Count the longest number
    features['longest_number'] = get_longest_number_string_ratio(query)
    features['attack'] = attack
    return features


def extract_features_reduced(query, attack):
    """
    Extract the features for a DNS query string
    The features are:
        - Number of alphanumeric characters in proportion to the query's length (alphanumeric: 0.8)
        - Longest consecutive number in the string in proportion to the query's length (longest_number: 0.1)
    """
    length = len(query)
    # Create dictionary to hold the values
    features = {'alphanumeric': 0, 'longest_number': 0}
    if length > 0:
        # Alphanumeric characters in query
        query_alphanumeric = list(filter(str.isalnum, query))
        # Count the number of repetitions of the alphanumeric characters
        features['alphanumeric'] = len(query_alphanumeric) / length
        # Feature that measures the longest string of numbers that are together
        features['longest_number'] = get_longest_number_string_ratio(query)
    features['attack'] = attack
    return features


def extract_features_entropy_and_ratios(query, attack):
    """
    Extract the features for a DNS query string
    The features are:
        - Letters ratio
        - Digits ratio
        - Entropy
        - Longest letters string
        - Longest digit string
        - Longest meaningful word
        - Symbols ratio
    Note: The features have the naming format "x_feature", where x is a number,
    to keep the previous feature order after they were renamed for consistency
    """
    features = {}
    features['attack'] = attack
    features['0_letters'] = get_letters_ratio(query)
    features['1_numbers'] = get_digits_ratio(query)
    features['2_entropy'] = metric_entropy(query)
    features['3_longest_letters'] = get_longest_letters_string_ratio(query)
    features['4_longest_number'] = get_longest_number_string_ratio(query)
    features['5_longest_meaningful'] = get_longest_meaningful_word_ratio(query)
    features['6_symbols'] = get_symbols_ratio(query)
    return features


"""
Main functions
TODO: Refactor to use 'parse_BRO_log_file'
"""


def create_feature_vector_from_log_file(infile, FV_function):
    """
    Open log file with DNS queries and create feature vector
    infile: log file
    FV_function: chosen function to create feature vector
    """
    slash_position = infile.rfind('/') # used in case the infile is a path to the file
    outfile = infile[:slash_position + 1] + "FV_" + infile[slash_position + 1:]

    feature_dictionary_list = []

    with open(infile) as inf, open(outfile, 'w') as outf:
        for row in csv.reader(inf, delimiter='\t'):
            if row and row[0][0] != '#':
                # Parse the IP and query from file
                IP = row[4].strip()
                query = row[9].split('.')[0]
                # Determine the attack tag and extract features for query
                attack = 1 if IP == '1.1.1.1' else 0
                features = FV_function(query, attack)
                outf.write("%s - %s | Features: %s\n" % (query, IP, features))
                # Append to features list
                feature_dictionary_list.append(features)
        # Create DataFrame from dictionary
        df = pd.DataFrame(feature_dictionary_list).fillna(0)

    return df


def create_feature_vector_from_log_file_tunnelling(infile, FV_function):
    """
    Open log file with DNS queries and create feature vector.
    Treats files with tunnelling data, where the attacks are directed to
    the domain 'test.com'.
    infile: log file
    FV_function: chosen function to create feature vector
    """
    slash_position = infile.rfind('/') # used in case the infile is a path to the file
    outfile = infile[:slash_position + 1] + "FV_" + infile[slash_position + 1:]

    feature_dictionary_list = []

    with open(infile) as inf, open(outfile, 'w') as outf:
        for row in csv.reader(inf, delimiter='\t'):
            if row and row[0][0] != '#':
                # Parse the domain and query from file
                try:
                    domain = row[9].split('.')[-2] + '.' + row[9].split('.')[-1]
                except IndexError:
                    domain = ''
                query = row[9].split('.')[0]
                # Determine the attack tag and extract features for query
                # The attacks are directed to the domain 'test.com'
                attack = 1 if domain == 'test.com' else 0
                features = FV_function(query, attack)
                outf.write("%s - %s | Features: %s\n" % (query, domain, features))
                # Append to features list
                feature_dictionary_list.append(features)
        # Create DataFrame from dictionary
        df = pd.DataFrame(feature_dictionary_list).fillna(0)

    return df
