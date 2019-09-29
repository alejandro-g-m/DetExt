import os
import sys
sys.path.append('..') # The code is in the parent directory
from collections import Counter
import unittest
import numpy as np
from feature_vector_creation import *


class TestHelperFunctions(unittest.TestCase):

    def test_get_char_ratio(self):
        # Test no chars
        np.testing.assert_almost_equal(get_char_ratio('12/_'), 0, decimal=3)
        # Test empty string
        np.testing.assert_almost_equal(get_char_ratio(''), 0, decimal=3)
        # Test several char strings with different length
        np.testing.assert_almost_equal(get_char_ratio('1a23bc1def2c'), 0.583, decimal=3)
        # Test a single char
        np.testing.assert_almost_equal(get_char_ratio('Z'), 1, decimal=3)
        # Test a long string of only chars
        np.testing.assert_almost_equal(get_char_ratio('aaabqoipuwerqllllllllsfdadfasdf'), 1, decimal=3)
        # Test different char strings separated by "other" characters
        np.testing.assert_almost_equal(get_char_ratio('a_bb*ccc/dddd-55555'), 0.526, decimal=3)
        # Test only "other" characters
        np.testing.assert_almost_equal(get_char_ratio(pt), 0, decimal=3)
        # Test a few chars separated by strings of "other" characters
        np.testing.assert_almost_equal(get_char_ratio('_/*n-+,.MM'), 0.3, decimal=3)
        # Test chars separated by digits and "other" characters
        np.testing.assert_almost_equal(get_char_ratio('1s_5s+4fa^sf!45'), 0.4, decimal=3)
        # Test several chars isolated
        np.testing.assert_almost_equal(get_char_ratio('a1b2c3d4'), 0.5, decimal=3)

    def test_get_digit_ratio(self):
        # Test no digits
        np.testing.assert_almost_equal(get_digit_ratio('aZB/_'), 0, decimal=3)
        # Test empty string
        np.testing.assert_almost_equal(get_digit_ratio(''), 0, decimal=3)
        # Test several digit strings with different length
        np.testing.assert_almost_equal(get_digit_ratio('1a23bcL0982c'), 0.583, decimal=3)
        # Test a single digit
        np.testing.assert_almost_equal(get_digit_ratio('0'), 1, decimal=3)
        # Test a long string of only digits
        np.testing.assert_almost_equal(get_digit_ratio('112223444444556677890000'), 1, decimal=3)
        # Test different digit strings separated by "other" characters
        np.testing.assert_almost_equal(get_digit_ratio('5_66*333/1111-lplm'), 0.555, decimal=3)
        # Test only "other" characters
        np.testing.assert_almost_equal(get_digit_ratio(pt), 0, decimal=3)
        # Test a few digits separated by strings of "other" characters
        np.testing.assert_almost_equal(get_digit_ratio('_/*3-+,.44'), 0.3, decimal=3)
        # Test digits separated by chars and "other" characters
        np.testing.assert_almost_equal(get_digit_ratio('1s_5s+40a^sf!45'), 0.4, decimal=3)
        # Test several digits isolated
        np.testing.assert_almost_equal(get_digit_ratio('a1b2c3d4'), 0.5, decimal=3)

    def test_get_symbols_ratio(self):
        # Test no symbols
        np.testing.assert_almost_equal(get_symbols_ratio('a1CbA90'), 0, decimal=3)
        # Test empty string
        np.testing.assert_almost_equal(get_symbols_ratio(''), 0, decimal=3)
        # Test several symbol strings with different length
        np.testing.assert_almost_equal(get_symbols_ratio('_a*?bcL=)&%c'), 0.583, decimal=3)
        # Test a single symbol
        np.testing.assert_almost_equal(get_symbols_ratio('-'), 1, decimal=3)
        # Test a long string of all punctuation characters
        np.testing.assert_almost_equal(get_symbols_ratio(pt), 1, decimal=3)
        # Test different symbol strings separated by digit and chars
        np.testing.assert_almost_equal(get_symbols_ratio(')6!#7+/·A^});P0391'), 0.555, decimal=3)
        # Test a few symbols separated by some digits
        np.testing.assert_almost_equal(get_symbols_ratio('_/*3-+,.44'), 0.7, decimal=3)
        # Test symbols separated by digits and chars
        np.testing.assert_almost_equal(get_symbols_ratio('1s_5s+40a^sf!45'), 0.266, decimal=3)
        # Test several symbols isolated
        np.testing.assert_almost_equal(get_symbols_ratio('a?b_c{d]'), 0.5, decimal=3)

    def test_get_longest_number_string_ratio(self):
        """
        get_longest_numer_string_ratio is a wrapper for get_longest_numer_string
        So both functions are tested here
        """
        # Test no numbers
        np.testing.assert_almost_equal(get_longest_number_string_ratio('abc'), 0, decimal=3)
        # Test empty string
        np.testing.assert_almost_equal(get_longest_number_string_ratio(''), 0, decimal=3)
        # Test several number strings with different length
        np.testing.assert_almost_equal(get_longest_number_string_ratio('a1b23c456de7f'), 0.230, decimal=3)
        # Test a single number
        np.testing.assert_almost_equal(get_longest_number_string_ratio('1'), 1, decimal=3)
        # Test a long string of only numbers
        np.testing.assert_almost_equal(get_longest_number_string_ratio('111111123456789990'), 1, decimal=3)
        # Test different number strings separated by "other" characters
        np.testing.assert_almost_equal(get_longest_number_string_ratio('1_22*333/4444-55555'), 0.263, decimal=3)
        # Test only "other" characters
        np.testing.assert_almost_equal(get_longest_number_string_ratio(pt), 0, decimal=3)
        # Test a few numbers separated by strings of "other" characters
        np.testing.assert_almost_equal(get_longest_number_string_ratio('_/*9-+,.00'), 0.2, decimal=3)
        # Test numbers separated by letters and "other" characters
        np.testing.assert_almost_equal(get_longest_number_string_ratio('_ab*c/defg1234567890ab1c2d3g4r5t6u7i89l000'), 0.238, decimal=3)
        # Test several numbers isolated
        np.testing.assert_almost_equal(get_longest_number_string_ratio('a1b2c3d4'), 0.125, decimal=3)

    def test_get_longest_char_string_ratio(self):
        """
        get_longest_char_string_ratio is a wrapper for get_longest_char_string
        So both functions are tested here
        """
        # Test no chars
        np.testing.assert_almost_equal(get_longest_char_string_ratio('12/_'), 0, decimal=3)
        # Test empty string
        np.testing.assert_almost_equal(get_longest_char_string_ratio(''), 0, decimal=3)
        # Test several char strings with different length
        np.testing.assert_almost_equal(get_longest_char_string_ratio('1a23bc1def2c'), 0.25, decimal=3)
        # Test a single char
        np.testing.assert_almost_equal(get_longest_char_string_ratio('Z'), 1, decimal=3)
        # Test a long string of only chars
        np.testing.assert_almost_equal(get_longest_char_string_ratio('aaabqoipuwerqllllllllsfdadfasdf'), 1, decimal=3)
        # Test different char strings separated by "other" characters
        np.testing.assert_almost_equal(get_longest_char_string_ratio('a_bb*ccc/dddd-55555'), 0.21, decimal=3)
        # Test only "other" characters
        np.testing.assert_almost_equal(get_longest_char_string_ratio(pt), 0, decimal=3)
        # Test a few chars separated by strings of "other" characters
        np.testing.assert_almost_equal(get_longest_char_string_ratio('_/*n-+,.MM'), 0.2, decimal=3)
        # Test chars separated by digits and "other" characters
        np.testing.assert_almost_equal(get_longest_char_string_ratio('1s_5s+4fa^sf!45'), 0.133, decimal=3)
        # Test several chars isolated
        np.testing.assert_almost_equal(get_longest_char_string_ratio('a1b2c3d4'), 0.125, decimal=3)

    def test_get_all_substrings(self):
        # Test empty string
        self.assertEqual(get_all_substrings(''), [])
        # Test string
        self.assertEqual(Counter(get_all_substrings('abcd')),
        Counter(['abcd', 'a', 'b', 'c', 'd', 'ab', 'abc', 'bc', 'bcd', 'cd']))
        # Test one char string
        self.assertEqual(get_all_substrings('Z'), ['Z'])
        # Test digits string
        self.assertEqual(Counter(get_all_substrings('123')),
        Counter(['1', '2', '3', '12', '23', '123']))
        # Test string with the same character repeated
        self.assertEqual(Counter(get_all_substrings('LLLL')),
        Counter(['L', 'LL', 'LLL', 'LLLL', 'L', 'LL', 'LLL', 'L', 'LL', 'L']))
        # Test another string
        self.assertEqual(Counter(get_all_substrings('aA')),
        Counter(['a', 'A', 'aA']))

    def test_has_digits_or_punctuation(self):
        # Test empty string
        self.assertFalse(has_digits_or_punctuation(''))
        # Test character string
        self.assertFalse(has_digits_or_punctuation(al))
        # Test digit string
        self.assertTrue(has_digits_or_punctuation(dg))
        # Test punctuation string
        self.assertTrue(has_digits_or_punctuation(pt))
        # Test one digit
        self.assertTrue(has_digits_or_punctuation('1'))
        # Test one symbol
        self.assertTrue(has_digits_or_punctuation('+'))
        # Test one char
        self.assertFalse(has_digits_or_punctuation('A'))
        # Test several chars lower and upper case
        self.assertFalse(has_digits_or_punctuation('aAzZjfurADSFI'))
        # Test digits and symbols
        self.assertTrue(has_digits_or_punctuation('8¿)/*445#@970'))

    def test_get_longest_meaningful_word(self):
        # Test empty string
        self.assertEqual(get_longest_meaningful_word(''), '')
        # Test one letter string
        self.assertEqual(get_longest_meaningful_word('a'), 'a')
        # Test one letter string
        self.assertEqual(get_longest_meaningful_word('z'), 'z')
        # Test one letter string
        self.assertEqual(get_longest_meaningful_word('X'), 'X')
        # Test one letter string
        self.assertEqual(get_longest_meaningful_word('L'), 'L')
        # Test low and upper case
        self.assertEqual(get_longest_meaningful_word('hEllO'), 'hEllO')
        # Test digits and symbols
        self.assertEqual(get_longest_meaningful_word(dg + pt), '')
        # Test long string with no meaningful words
        self.assertEqual(len(get_longest_meaningful_word('ojiLpYBrhs')), 1)
        # Test mix of chars, digits and symbols
        self.assertEqual(get_longest_meaningful_word('j/9486795674568778576L*/-microscopic894*helloB#'), 'microscopic')
        # Test several words
        self.assertEqual(get_longest_meaningful_word('potatoskydrawerbookchairtablemousekeyboard'), 'keyboard')

    def test_get_longest_meaningful_word_ratio_ratio(self):
        # Test empty string
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio(''), 0, decimal=3)
        # Test one letter string
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('a'), 1, decimal=3)
        # Test one letter string
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('z'), 1, decimal=3)
        # Test one letter string
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('X'), 1, decimal=3)
        # Test one letter string
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('L'), 1, decimal=3)
        # Test low and upper case
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('hEllO'), 1, decimal=3)
        # Test digits and symbols
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio(dg + pt), 0, decimal=3)
        # Test long string with no meaningful words
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('ojiLpYBrhs'), 0.1, decimal=3)
        # Test mix of chars, digits and symbols
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('j/9486795674568778576L*/-microscopic894*helloB#'), 0.234, decimal=3)
        # Test several words
        np.testing.assert_almost_equal(get_longest_meaningful_word_ratio('potatoskydrawerbookchairtablemousekeyboard'), 0.19, decimal=3)

    def test_metric_entropy(self):
        """
        Tested using http://www.shannonentropy.netmark.pl/
        All the characters in the test string should exist in url_characters
        range_url is also tested here
        """
        # Test empty string
        np.testing.assert_almost_equal(metric_entropy(''), 0, decimal=4)
        # Test word
        np.testing.assert_almost_equal(metric_entropy('hello'), 0.38439, decimal=4)
        # Test string
        np.testing.assert_almost_equal(metric_entropy('sample_text'), 0.28144, decimal=4)
        # Test same digit
        np.testing.assert_almost_equal(metric_entropy('111'), 0, decimal=4)
        # Test same char
        np.testing.assert_almost_equal(metric_entropy('ZZZZZZZZZZZ'), 0, decimal=4)
        # Test same symbol
        np.testing.assert_almost_equal(metric_entropy('$$$$$$'), 0, decimal=4)
        # Test chars, digits and symbols
        np.testing.assert_almost_equal(metric_entropy('aAzZ109*-(_'), 0.31449, decimal=4)
        # Test chars, digits and symbols
        np.testing.assert_almost_equal(metric_entropy('a1bc5698!+78lo'), 0.26175, decimal=4)
        # Test all possible characters in url
        np.testing.assert_almost_equal(metric_entropy(url_characters), 0.08569, decimal=4)
        # Test lower and upper case chars
        np.testing.assert_almost_equal(metric_entropy('aLopiDFDJuyudnDMBUs'), 0.20486, decimal=4)



class TestFeatureVectorCreation(unittest.TestCase):

    def test_extract_features_with_letter_counting(self):
        """
        The function "extract_features_with_letter_counting" returns a dictionary.
        The tests consist in checking that the returned dictionary has the same
        keys as expected and that the values of the keys are correct.
        """
        empty = {x:0 for x in al + dg}
        # Test a simple string of letters
        desired = empty.copy()
        desired.update({'a':0.333, 'b':0.333, 'c':0.333, 'other':0, 'longest_number':0, 'attack':0})
        test = extract_features_with_letter_counting('abc', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of numbers
        desired = empty.copy()
        desired.update({'1':1, 'other':0, 'longest_number':1, 'attack':0})
        test = extract_features_with_letter_counting('111', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of "other" characters
        desired = empty.copy()
        desired.update({'other':1, 'longest_number':0, 'attack':0})
        test = extract_features_with_letter_counting('_*-+¿.(%#', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test number and "other"
        desired = empty.copy()
        desired.update({'z':0.333, '0':0.333, 'other':0.333, 'longest_number':0.333, 'attack':1})
        test = extract_features_with_letter_counting('z0*', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all letters
        desired = empty.copy()
        desired.update({'a':0.0384, 'b':0.0384, 'c':0.0384, 'd':0.0384, 'e':0.0384,
        'f':0.0384, 'g':0.0384, 'h':0.0384, 'i':0.0384, 'j':0.0384, 'k':0.0384,
        'l':0.0384, 'm':0.0384, 'n':0.0384, 'o':0.0384, 'p':0.0384, 'q':0.0384,
        'r':0.0384, 's':0.0384, 't':0.0384, 'u':0.0384, 'v':0.0384, 'w':0.0384,
        'x':0.0384, 'y':0.0384, 'z':0.0384, 'other':0, 'longest_number':0, 'attack':1})
        test = extract_features_with_letter_counting('abcdefghijklmnopqrstuvwxyz', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all numbers
        desired = empty.copy()
        desired.update({'0':0.1, '1':0.1, '2':0.1, '3':0.1, '4':0.1, '5':0.1, '6':0.1,
        '7':0.1, '8':0.1, '9':0.1, 'other':0, 'longest_number':1, 'attack':1})
        test = extract_features_with_letter_counting('3498501267', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test empty string
        desired = empty.copy()
        desired.update({'other':0, 'longest_number':0, 'attack':1})
        test = extract_features_with_letter_counting('', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test long combined string
        desired = empty.copy()
        desired.update({'a':0.0454, 'b':0.0454, 'c':0.0454, 'd':0.0454, 'e':0.0227,
        'f':0.0227, 'g':0.0454, 'i':0.0227, 'l':0.0227, 'r':0.0227, 't':0.0227,
        'u':0.0227, '0':0.0909, '1':0.0454, '2':0.0454, '3':0.0454, '4':0.0454,
        '5':0.0454, '6':0.0454, '7':0.0454, '8':0.0681, '9':0.0227, 'other':0.1136,
        'longest_number':0.2272, 'attack':1})
        test = extract_features_with_letter_counting('_ab*c/defg1234567890ab1c2¿d3g4-r5t6u7i88l000', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)

    def test_extract_features_with_letters_and_numbers(self):
        """
        The function "extract_features_with_letters_and_numbers" returns a dictionary.
        The tests consist in checking that the returned dictionary has the same
        keys as expected and that the values of the keys are correct.
        """
        # Test a simple string of letters
        desired = {'letters':1, 'numbers':0, 'other':0, 'longest_number':0, 'attack':0}
        test = extract_features_with_letters_and_numbers('abc', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of numbers
        desired = {'letters':0, 'numbers':1, 'other':0, 'longest_number':1, 'attack':0}
        test = extract_features_with_letters_and_numbers('111', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of "other" characters
        desired = {'letters':0, 'numbers':0, 'other':1, 'longest_number':0, 'attack':0}
        test = extract_features_with_letters_and_numbers('_*-+¿.(%#', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a capital lettter, number and "other"
        desired = {'letters':0.333, 'numbers':0.333, 'other':0.333, 'longest_number':0.333, 'attack':1}
        test = extract_features_with_letters_and_numbers('Z0*', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all letters
        desired = {'letters':1, 'numbers':0, 'other':0, 'longest_number':0, 'attack':1}
        test = extract_features_with_letters_and_numbers('abcdefghijklmnopqrstuvwxyz', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all numbers
        desired = {'letters':0, 'numbers':1, 'other':0, 'longest_number':1, 'attack':1}
        test = extract_features_with_letters_and_numbers('3498501267', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test empty string
        desired = {'letters':0, 'numbers':0, 'other':0, 'longest_number':0, 'attack':1}
        test = extract_features_with_letters_and_numbers('', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test long combined string
        desired = {'letters':0.3863, 'numbers':0.5, 'other':0.1136, 'longest_number':0.2272, 'attack':1}
        test = extract_features_with_letters_and_numbers('_ab*c/defg1234567890ab1c2¿d3g4-r5t6u7i88l000', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)

    def test_extract_features_reduced(self):
        """
        The function "extract_features_reduced" returns a dictionary.
        The tests consist in checking that the returned dictionary has the same
        keys as expected and that the values of the keys are correct.
        """
        # Test a simple string of letters
        desired = {'alphanumeric':1, 'longest_number':0, 'attack':0}
        test = extract_features_reduced('abc', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of numbers
        desired = {'alphanumeric':1, 'longest_number':1, 'attack':0}
        test = extract_features_reduced('111', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of "other" characters
        desired = {'alphanumeric':0, 'longest_number':0, 'attack':0}
        test = extract_features_reduced('_*-+¿.(%#', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a capital lettter, number and "other"
        desired = {'alphanumeric':0.666, 'longest_number':0.333, 'attack':1}
        test = extract_features_reduced('Z0*', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all letters
        desired = {'alphanumeric':1, 'longest_number':0, 'attack':1}
        test = extract_features_reduced('abcdefghijklmnopqrstuvwxyz', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all numbers
        desired = {'alphanumeric':1, 'longest_number':1, 'attack':1}
        test = extract_features_reduced('3498501267', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test empty string
        desired = {'alphanumeric':0, 'longest_number':0, 'attack':1}
        test = extract_features_reduced('', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test long combined string
        desired = {'alphanumeric':0.8863, 'longest_number':0.2272, 'attack':1}
        test = extract_features_reduced('_ab*c/defg1234567890ab1c2¿d3g4-r5t6u7i88l000', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)

    def test_extract_features_entropy_and_ratios(self):
        """
        The function "extract_features_entropy_and_ratios" returns a dictionary.
        The tests consist in checking that the returned dictionary has the same
        keys as expected and that the values of the keys are correct.
        """
        # Test empty string
        desired = {'chars':0, 'digits':0, 'symbols':0, 'longest_chars':0,
        'longest_digits':0, 'entropy':0, 'longest_meaningful':0, 'attack':0}
        test = extract_features_entropy_and_ratios('', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of letters
        desired = {'chars':1, 'digits':0, 'symbols':0, 'longest_chars':1,
        'longest_digits':0, 'entropy':0.52832, 'longest_meaningful':0.666, 'attack':0}
        test = extract_features_entropy_and_ratios('abc', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of numbers
        desired = {'chars':0, 'digits':1, 'symbols':0, 'longest_chars':0,
        'longest_digits':1, 'entropy':0, 'longest_meaningful':0, 'attack':0}
        test = extract_features_entropy_and_ratios('111', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a simple string of symbols
        desired = {'chars':0, 'digits':0, 'symbols':1, 'longest_chars':0,
        'longest_digits':0, 'entropy':0.43083, 'longest_meaningful':0, 'attack':0}
        test = extract_features_entropy_and_ratios('_*-+!(', 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test a capital lettter, number and "other"
        desired = {'chars':0.333, 'digits':0.333, 'symbols':0.333, 'longest_chars':0.333,
        'longest_digits':0.333, 'entropy':0.52832, 'longest_meaningful':0.333, 'attack':1}
        test = extract_features_entropy_and_ratios('Z0*', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all letters
        desired = {'chars':1, 'digits':0, 'symbols':0, 'longest_chars':1,
        'longest_digits':0, 'entropy':0.18079, 'longest_meaningful':0.115, 'attack':1}
        test = extract_features_entropy_and_ratios(al, 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all numbers
        desired = {'chars':0, 'digits':1, 'symbols':0, 'longest_chars':0,
        'longest_digits':1, 'entropy':0.33219, 'longest_meaningful':0, 'attack':1}
        test = extract_features_entropy_and_ratios(dg, 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all symbols in url
        desired = {'chars':0, 'digits':0, 'symbols':1, 'longest_chars':0,
        'longest_digits':0, 'entropy':0.33219, 'longest_meaningful':0, 'attack':1}
        test = extract_features_entropy_and_ratios("$-_+!*'(),", 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test all characters in url
        desired = {'chars':0.722, 'digits':0.138, 'symbols':0.138, 'longest_chars':0.722,
        'longest_digits':0.138, 'entropy':0.08569, 'longest_meaningful':0.041, 'attack':0}
        test = extract_features_entropy_and_ratios(url_characters, 0)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)
        # Test long combined string
        desired = {'chars':0.3863, 'digits':0.5, 'symbols':0.1136, 'longest_chars':0.09,
        'longest_digits':0.2272, 'entropy':0.10509, 'longest_meaningful':0.068, 'attack':1}
        test = extract_features_entropy_and_ratios('_ab*c+defg1234567890ab1c2)d3g4-r5t6u7i88l000', 1)
        self.assertCountEqual(desired.keys(), test.keys())
        for key, value in test.items():
            np.testing.assert_almost_equal(value, desired[key], decimal=3)

class TestFeatureVectorFromLogFile(unittest.TestCase):

    def setUp(self):
        self.input_file = 'test_logs'
        with open(self.input_file, 'w') as outf:
            outf.write("1530627176.803513	CAyYP92af1ISQvv9sk	192.168.0.27	"
            "40183	8.8.4.4	53	udp	30685	-	test1.edu/	1	C_INTERNET	1	A	"
            "3	NXDOMAIN	F	F	T	F	0	-	-	F")
            outf.write("\n1530627176.803513	C1FHz61LMAW6GylnT1	192.168.0.27	"
            "52003	8.8.4.4	53	udp	28639	0.224005	*.google.com	1	"
            "C_INTERNET	1	A	0	NOERROR	F	F	T	T	0	216.58.209.206	"
            "299.000000	F")
            outf.write("\n1530627178.943508	CTDAu34MfRAurSLQO4	192.168.0.27	"
            "39176	1.1.1.1	53	udp	10310	0.468023	a0_.google.com	1	"
            "C_INTERNET	1	A	0	NOERROR	F	F	T	T	0	"
            "video.l.google.com,216.58.209.206	21593.000000,293.000000	F")

    def test_create_feature_vector_from_log_file(self):
        test = create_feature_vector_from_log_file(self.input_file, extract_features_reduced)
        desired = np.array([[1, 0, 0.2], [0, 0, 0], [0.666, 1, 0.333]])
        np.testing.assert_array_almost_equal(test, desired, decimal=3)

    def tearDown(self):
        os.remove(self.input_file)
        os.remove('FV_' + self.input_file)


if __name__ == '__main__':
    unittest.main()
