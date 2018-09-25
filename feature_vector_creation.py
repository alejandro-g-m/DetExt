import sys
import csv
import collections
import string
import re
import pandas as pd


"""
Helper Functions and Classes
"""


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


def get_longest_string_number(original_string):
    """
    Get the longest string of consecutive numbers in a string
    For example in "a1b23c456de7f" it would return "456"
    """
    longest_number_string = ''
    regex = r'([1-9]+)'
    matches = re.findall(regex, original_string)
    if matches:
        longest_number_string = max(matches, key=len)
    return longest_number_string


"""
Functions to create features vectors. Each function creates a different type of vector.
Several functions are created to check the performance of different features vectors.
"""


def extract_features_with_letter_counting(query, attack):
    """
    Extract the features for a DNS query string counting all the letters in the string
    The features are:
        - Count of alphanumeric characters (a: 1, b: 5, c: 2...)
        - Number of non-alphanumeric characters (other: 3)
        - Longest consecutive number in the string (longest_number: 2)
    """
    # Alphanumeric characters in query
    query_alphanumeric = list(filter(str.isalnum, query))
    # Non-alphanumeric characters in query
    query_non_alphanumeric = list(filter(lambda ch: not ch.isalnum(), query))
    # Create dictionary with the number of repetitions of the alphanumeric characters
    letters = collections.Counter(query_alphanumeric)
    if query_non_alphanumeric:
        # Feature that measures the repetitions of other characters
        letters['other'] = len(query_non_alphanumeric)
    # Feature that measures the longest string of numbers that are together
    longest_number_in_query = get_longest_string_number(query)
    letters['longest_number'] = len(longest_number_in_query)
    letters['attack'] = attack
    return letters


def extract_features_reduced(query, attack):
    """
    Extract the features for a DNS query string
    The features are:
        - Number of alphanumeric characters in proportion to the query's length (alphanumeric: 0.8)
        - Longest consecutive number in the string in proportion to the query's length (longest_number: 0.1)
    """
    # Alphanumeric characters in query
    query_alphanumeric = list(filter(str.isalnum, query))
    # Create dictionary to hold the values
    letters = {}
    # Count the number of repetitions of the alphanumeric characters
    letters['alphanumeric'] = len(query_alphanumeric) / len(query)
    # Feature that measures the longest string of numbers that are together
    longest_number_in_query = get_longest_string_number(query)
    letters['longest_number'] = len(longest_number_in_query) / len(query)
    letters['attack'] = attack
    return letters


"""
Main function
"""


def create_feature_vector_from_log_file(infile, FV_function):
    """
    Open log file with DNS queries and create feature vector
    infile: log file
    FV_function: chosen function to create features vector
    """
    outfile = "FV_" + infile

    feature_dictionary_list = []

    with open(infile) as inf, open(outfile, 'w') as outf:
        for row in csv.reader(inf, delimiter='\t'):
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
