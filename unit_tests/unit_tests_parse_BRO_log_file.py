import unittest
import os
import sys
sys.path.append('..') # The code is in the parent directory
from parse_BRO_log_file import *


class TestBROParser(unittest.TestCase):

    def setUp(self):
        self.input_file = 'test_logs'
        # Sample log file with three lines
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
        self.df = BRO_DNS_record.parse_BRO_log_file(self.input_file)

    def test_parse_BRO_log_file(self):
        """
        Core functionality of TestBROParse, these are the main tests.
        """
        # Test values of first line
        self.assertEqual('1530627176.803513', self.df.iloc[0].ts)
        self.assertEqual('CAyYP92af1ISQvv9sk', self.df.iloc[0].uid)
        self.assertEqual('192.168.0.27', self.df.iloc[0].origin_IP)
        self.assertEqual('40183', self.df.iloc[0].origin_port)
        self.assertEqual('8.8.4.4', self.df.iloc[0].dest_IP)
        self.assertEqual('53', self.df.iloc[0].dest_port)
        self.assertEqual('udp', self.df.iloc[0].protocol)
        self.assertEqual('30685', self.df.iloc[0].trans_ID)
        self.assertEqual('-', self.df.iloc[0].round_trip_time)
        self.assertEqual('test1.edu/', self.df.iloc[0].query)
        self.assertEqual('1', self.df.iloc[0].qclass)
        self.assertEqual('C_INTERNET', self.df.iloc[0].qclass_name)
        self.assertEqual('1', self.df.iloc[0].query_type)
        self.assertEqual('A', self.df.iloc[0].query_type_name)
        self.assertEqual('3', self.df.iloc[0].response_code)
        self.assertEqual('NXDOMAIN', self.df.iloc[0].response_code_name)
        self.assertFalse(self.df.iloc[0].authoritative_answer)
        self.assertFalse(self.df.iloc[0].truncated_message)
        self.assertTrue(self.df.iloc[0].recursion_desired)
        self.assertFalse(self.df.iloc[0].recursion_available)
        self.assertEqual('0', self.df.iloc[0].Z)
        self.assertEqual(['-'], self.df.iloc[0].answers)
        self.assertEqual(['-'], self.df.iloc[0].caching_intervals)
        self.assertFalse(self.df.iloc[0].rejected)
        self.assertIsNone(self.df.iloc[0].total_answers)
        self.assertIsNone(self.df.iloc[0].total_replies)
        self.assertIsNone(self.df.iloc[0].saw_query)
        self.assertIsNone(self.df.iloc[0].saw_reply)
        self.assertIsNone(self.df.iloc[0].auth)
        self.assertIsNone(self.df.iloc[0].additional_responses)

        # Test values of second line
        self.assertEqual('1530627176.803513', self.df.iloc[1].ts)
        self.assertEqual('C1FHz61LMAW6GylnT1', self.df.iloc[1].uid)
        self.assertEqual('192.168.0.27', self.df.iloc[1].origin_IP)
        self.assertEqual('52003', self.df.iloc[1].origin_port)
        self.assertEqual('8.8.4.4', self.df.iloc[1].dest_IP)
        self.assertEqual('53', self.df.iloc[1].dest_port)
        self.assertEqual('udp', self.df.iloc[1].protocol)
        self.assertEqual('28639', self.df.iloc[1].trans_ID)
        self.assertEqual('0.224005', self.df.iloc[1].round_trip_time)
        self.assertEqual('*.google.com', self.df.iloc[1].query)
        self.assertEqual('1', self.df.iloc[1].qclass)
        self.assertEqual('C_INTERNET', self.df.iloc[1].qclass_name)
        self.assertEqual('1', self.df.iloc[1].query_type)
        self.assertEqual('A', self.df.iloc[1].query_type_name)
        self.assertEqual('0', self.df.iloc[1].response_code)
        self.assertEqual('NOERROR', self.df.iloc[1].response_code_name)
        self.assertFalse(self.df.iloc[1].authoritative_answer)
        self.assertFalse(self.df.iloc[1].truncated_message)
        self.assertTrue(self.df.iloc[1].recursion_desired)
        self.assertTrue(self.df.iloc[1].recursion_available)
        self.assertEqual('0', self.df.iloc[1].Z)
        self.assertEqual(['216.58.209.206'], self.df.iloc[1].answers)
        self.assertEqual(['299.000000'], self.df.iloc[1].caching_intervals)
        self.assertFalse(self.df.iloc[1].rejected)
        self.assertIsNone(self.df.iloc[1].total_answers)
        self.assertIsNone(self.df.iloc[1].total_replies)
        self.assertIsNone(self.df.iloc[1].saw_query)
        self.assertIsNone(self.df.iloc[1].saw_reply)
        self.assertIsNone(self.df.iloc[1].auth)
        self.assertIsNone(self.df.iloc[1].additional_responses)

        # Test values of third line
        self.assertEqual('1530627178.943508', self.df.iloc[2].ts)
        self.assertEqual('CTDAu34MfRAurSLQO4', self.df.iloc[2].uid)
        self.assertEqual('192.168.0.27', self.df.iloc[2].origin_IP)
        self.assertEqual('39176', self.df.iloc[2].origin_port)
        self.assertEqual('1.1.1.1', self.df.iloc[2].dest_IP)
        self.assertEqual('53', self.df.iloc[2].dest_port)
        self.assertEqual('udp', self.df.iloc[2].protocol)
        self.assertEqual('10310', self.df.iloc[2].trans_ID)
        self.assertEqual('0.468023', self.df.iloc[2].round_trip_time)
        self.assertEqual('a0_.google.com', self.df.iloc[2].query)
        self.assertEqual('1', self.df.iloc[2].qclass)
        self.assertEqual('C_INTERNET', self.df.iloc[2].qclass_name)
        self.assertEqual('1', self.df.iloc[2].query_type)
        self.assertEqual('A', self.df.iloc[2].query_type_name)
        self.assertEqual('0', self.df.iloc[2].response_code)
        self.assertEqual('NOERROR', self.df.iloc[2].response_code_name)
        self.assertFalse(self.df.iloc[2].authoritative_answer)
        self.assertFalse(self.df.iloc[2].truncated_message)
        self.assertTrue(self.df.iloc[2].recursion_desired)
        self.assertTrue(self.df.iloc[2].recursion_available)
        self.assertEqual('0', self.df.iloc[2].Z)
        self.assertEqual(['video.l.google.com', '216.58.209.206'], self.df.iloc[2].answers)
        self.assertEqual(['21593.000000', '293.000000'], self.df.iloc[2].caching_intervals)
        self.assertFalse(self.df.iloc[2].rejected)
        self.assertIsNone(self.df.iloc[2].total_answers)
        self.assertIsNone(self.df.iloc[2].total_replies)
        self.assertIsNone(self.df.iloc[2].saw_query)
        self.assertIsNone(self.df.iloc[2].saw_reply)
        self.assertIsNone(self.df.iloc[2].auth)
        self.assertIsNone(self.df.iloc[2].additional_responses)

    def test_get_not_A_records(self):
        # All the sample records are of type A
        self.assertTrue(BRO_DNS_record.get_not_A_records(self.df, True).\
            startswith('Empty DataFrame'))
        os.remove('not_A.txt')

    def tearDown(self):
        os.remove(self.input_file)



class TestBROParserCornerCases(unittest.TestCase):

    def test_parse_BRO_log_only_comments_file(self):
        input_file = 'test_logs'
        # Sample log file with only comment lines
        with open(input_file, 'w') as outf:
            outf.write("# whatever")
            outf.write("\n# this is just a comment")
            outf.write("\n# this is not the line you are looking for")
        df = BRO_DNS_record.parse_BRO_log_file(input_file)
        os.remove(input_file)
        self.assertTrue(df.empty)

    def test_parse_BRO_log_empty_file(self):
        input_file = 'test_logs'
        # Sample empty file
        with open(input_file, 'w') as outf:
            outf.write("")
        df = BRO_DNS_record.parse_BRO_log_file(input_file)
        os.remove(input_file)
        self.assertTrue(df.empty)

    def test_parse_BRO_log_file_wrong_file(self):
        # File does not exist
        self.assertRaises(FileNotFoundError,
            BRO_DNS_record.parse_BRO_log_file, 'wrong_file')



class TestBROParserUtils(unittest.TestCase):

    def test_fetch_field(self):
        """
        This function is already tested in 'test_parse_BRO_log_file' so this
        is just a basic test.
        """
        sample_row = ['first', 'second']
        # Happy path
        self.assertEqual('first', BRO_DNS_record.fetch_field(sample_row, 0))
        self.assertEqual('second', BRO_DNS_record.fetch_field(sample_row, 1))
        # Non existing field
        self.assertIsNone(BRO_DNS_record.fetch_field(sample_row, 2))



if __name__ == '__main__':
    unittest.main()
