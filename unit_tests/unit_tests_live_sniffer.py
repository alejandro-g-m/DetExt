import unittest
from unittest.mock import patch
import sys
sys.path.append('..') # The code is in the parent directory
import os
import glob
import logging
import threading
from sklearn.externals import joblib
from scapy.all import *
import live_sniffer as ls


NET_INTERFACE = os.getenv('TEST_NET_INTERFACE', 'wlp4s0')
LIVE_SNIFFER_PATH = os.path.dirname(os.path.abspath(ls.__file__))


def save_pcap_packet():
    """
    Helper function to save a PCAP packet for testing purposes.
    """
    def save_pcap(packet):
        if IP in packet and packet.haslayer(DNS) and packet.getlayer(DNS).qr == 0:
            wrpcap('temp.cap', packet) # Saves PCAP file
    # It will sniff 100 packets to try to find a valid packet and save it
    sniff(iface=NET_INTERFACE, filter='port 53', count=100, prn=save_pcap, store=False)


class TestLiveSnifferFunctions(unittest.TestCase):

    def setUp(self):
        ls.initial_setup()

    def test_initial_setup(self):
        """
        Test global variables used for the live sniffer processing.
        """
        # DEFAULT_MODEL
        self.assertEqual(f'{LIVE_SNIFFER_PATH}/models/'
                            'extract_features_entropy_and_ratios/'
                            'decision_trees+knn.pkl', ls.DEFAULT_MODEL)
        # FV_FUNCTION
        self.assertEqual(ls.extract_features_entropy_and_ratios, ls.FV_FUNCTION)
        # AVAILABLE_MODELS
        self.assertTrue(len(ls.AVAILABLE_MODELS) > 0)
        for mod in glob(f'{LIVE_SNIFFER_PATH}/models/'
                        'extract_features_entropy_and_ratios/*.pkl'):
            self.assertTrue(mod in ls.AVAILABLE_MODELS)

    @patch('sys.stdout') # This is mocked just so no output is shown
    def test_query_sniff_wrapper(self, _):
        """
        'query_sniff_wrapper' is a complex function to be tested. Maybe it
        should be divided in small chunks. This a basic test to check that
        it can be called without errors.

        Most of the functionality consists on decoding the DNS query, applying
        the machine learning functions that have been developed and showing the
        result to the user.

        Sample PCAP files are used to run the function.
        """
        model = ls.AVAILABLE_MODELS[-1]
        # Test regular DNS packet
        ls.query_sniff_wrapper(joblib.load(model))\
            (rdpcap('sample_happy_path.cap')[0])
        # Test packet not encoded with utf-8
        ls.query_sniff_wrapper(joblib.load(model))\
            (rdpcap('sample_encoding.cap')[0])

    @patch('sys.stdout') # This is mocked just so no output is shown
    @patch('builtins.input', create=True)
    def test_generate_menu_happy_path(self, mocked_input, _):
        """
        Test menu generation with simulated valid user input.
        """
        expected_model = len(ls.AVAILABLE_MODELS) - 1
        mocked_input.side_effect = ['', str(expected_model)]
        interface, model = ls.generate_menu()
        self.assertEqual('wlan0', interface)
        self.assertEqual(ls.AVAILABLE_MODELS[expected_model], model)

    @patch('sys.stdout') # This is mocked just so no output is shown
    @patch('builtins.input', create=True)
    def test_generate_menu_wrong_model(self, mocked_input, _):
        """
        Test menu generation with simulated invalid user input (wrong model).
        """
        wrong_model = len(ls.AVAILABLE_MODELS) + 1
        mocked_input.side_effect = ['random_interface', str(wrong_model)]
        # Used for logging test
        ls_logger = ls.logger.name
        with self.assertLogs(ls_logger, level='WARNING') as cm:
            interface, model = ls.generate_menu()
        # Test interface
        self.assertEqual('random_interface', interface)
        # Test model
        self.assertEqual(ls.DEFAULT_MODEL, model)
        # Test that the generated logs are correct
        self.assertEqual(cm.output,
            [f'WARNING:{ls_logger}:The selected model does not exist, '
            'using default model.'])

    def test_get_features_for_query(self):
        """
        'get_features_for_query' is used by other functions
        that are tested already, so no specific test is created.
        """
        pass

    def test_get_model_name(self):
        # Get all models in the 'models' folder
        for mod in glob(f'{LIVE_SNIFFER_PATH}/models/*/*.pkl'):
            # Extract the model's name
            expected_model_name = os.path.splitext(os.path.basename(mod))[0]
            # Assert the extracted name with the result of 'get_model_name'
            self.assertEqual(expected_model_name, ls.get_model_name(mod))

    def test_check_queries_in_models(self):
        """
        This test assumes that all of the models perform perfectly with the
        provided queries. They should classify properly the attack and the
        legitimate query. Otherwise, this test will fail.
        """
        queries = ['thisisnot.an.attack', 'lwnuehe666386336373138636137.this.is']
        queries_in_models = ls.check_queries_in_models(queries)
        # Check both queries are present
        for query in queries:
            self.assertTrue(query in queries)
        # Check that the models only classify as an attack the malicious query
        for model in ls.AVAILABLE_MODELS:
            mname = ls.get_model_name(model)
            self.assertFalse(mname in queries_in_models['thisisnot.an.attack'])
            self.assertTrue(mname in
                queries_in_models['lwnuehe666386336373138636137.this.is'])

    def test_print_queries_in_models(self):
        """
        This is not a complete test, but 'print_queries_in_models' is just
        a helper function. As the previous test, relies on a perfect performance
        for the given queries.
        """
        queries = ['thisisnot.an.attack', 'lwnuehe666386336373138636137.this.is']
        print_result = ls.print_queries_in_models(ls.check_queries_in_models(queries))
        # Count that each query appears just once in the print_result
        for query in queries:
            self.assertEqual(1, print_result.count(query))
        # Count that each model appears just once in the print result
        # '\n' is needed so only the exact model is found
        for model in ls.AVAILABLE_MODELS:
            self.assertEqual(1, print_result.count(f'\n{ls.get_model_name(model)}\n'))



class TestLiveSnifferMain(unittest.TestCase):

    @patch('sys.stdout') # This is mocked just so no output is shown
    @patch('builtins.input', create=True)
    def test_main_wrong_interface(self, mocked_input, _):
        """
        Test main run with simulated invalid user input (wrong interface).
        """
        interface = 'wrong_interface'
        model = len(ls.AVAILABLE_MODELS) - 1
        mocked_input.side_effect = [interface, str(model)]
        ls_logger = ls.logger.name
        # Logging test
        with self.assertLogs(ls_logger, level='ERROR') as cm:
            ls.main()
        # Test that the generated logs are correct
        self.assertEqual(cm.output,
            [f'ERROR:{ls_logger}:{interface} is not a valid interface!'])



if __name__ == '__main__':
    unittest.main()
