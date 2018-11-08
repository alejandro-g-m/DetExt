import unittest
import numpy as np
import os
from feature_vector_creation import *
from string import ascii_lowercase as al, digits as dg

class TestFeatureVectorCreation(unittest.TestCase):

    def test_get_longest_string_number(self):
        # Test no numbers
        self.assertEqual(get_longest_string_number('abc'), '')
        # Test empty string
        self.assertEqual(get_longest_string_number(''), '')
        # Test several number strings with different length
        self.assertEqual(get_longest_string_number('a1b23c456de7f'), '456')
        # Test a single number
        self.assertEqual(get_longest_string_number('1'), '1')
        # Test a long string of only numbers
        self.assertEqual(get_longest_string_number('111111123456789990'), '111111123456789990')
        # Test different number strings separated by "other" characters
        self.assertEqual(get_longest_string_number('1_22*333/4444-55555'), '55555')
        # Test only "other" characters
        self.assertEqual(get_longest_string_number('_/*-+,.'), '')
        # Test a few numbers separated by strings of "other" characters
        self.assertEqual(get_longest_string_number('_/*9-+,.00'), '00')
        # Test numbers separated by letters and "other" characters
        self.assertEqual(get_longest_string_number('_ab*c/defg1234567890ab1c2d3g4r5t6u7i89l000'), '1234567890')
        # Test several numbers isolated
        self.assertEqual(len(get_longest_string_number('a1b2c3d4')), 1)

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
        desired.update({'attack':1})
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
