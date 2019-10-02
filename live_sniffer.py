import sys
import glob
import logging
from sklearn.externals import joblib
from scapy.all import *
from feature_vector_creation import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initial_setup():
    # Declare global variables
    global DEFAULT_MODEL
    global FV_FUNCTION
    global AVAILABLE_MODELS
    # Load models using the feature vector with entropy and ratios
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_directory = f'{current_path}/models'
    function_directory = '/extract_features_entropy_and_ratios'
    DEFAULT_MODEL = model_directory + function_directory + '/decision_trees+knn.pkl'
    FV_FUNCTION = extract_features_entropy_and_ratios

    AVAILABLE_MODELS = sorted(
    [mod for mod in glob(model_directory + function_directory + '/*.pkl')])


def query_sniff_wrapper(model):
    """
    Wrapper function in order to pass the "model" parameter to query_sniff from
    the sniff function.
    """
    def query_sniff(packet):
        """
        Analyses a sniffed packet: if it has DNS information and the DNS query
        has at least 3 levels of subdomains the features for the query are
        extracted. Then, the chosen model decides if it is an attack or not.
        """
        if IP in packet:
            if packet.haslayer(DNS) and packet.getlayer(DNS).qr == 0:
                ip_src = packet[IP].src
                ip_dst = packet[IP].dst
                try:
                    # Try to decode the query in utf-8
                    query = packet.getlayer(DNS).qd.qname.decode('utf-8')
                    # Only process queries with at least 3 levels of subdomains
                    if len(query.split('.')) > 3:
                        subdomain = query.split('.')[0]
                        features = (pd.DataFrame(FV_FUNCTION(subdomain, -1),
                        index=[0]).fillna(0).drop('attack', 1))
                        if model.predict(features) == 0:
                            prediction = color.GREEN + "LEGIT" + color.END
                        else:
                            prediction = color.RED + "ATTACK" + color.END
                        print(str(ip_src) + " -> " + str(ip_dst) +
                        " : " + "(" + query + ")" + " : " + prediction)
                except UnicodeDecodeError:
                    # Handle other encodings
                    try:
                        query = packet.getlayer(DNS).qd.qname
                        # Only process queries with at least 3 levels of subdomains
                        if len(query.split(b'.')) > 3:
                            # Scapy returns a byte object. Let's obtain the raw
                            # data without decoding because the models are trained
                            # in this way
                            subdomain = str(query.split(b'.')[0])[2:-1]
                            features = (pd.DataFrame(FV_FUNCTION(subdomain, -1),
                            index=[0]).fillna(0).drop('attack', 1))
                            if model.predict(features) == 0:
                                prediction = color.GREEN + "LEGIT" + color.END
                            else:
                                prediction = color.RED + "ATTACK" + color.END
                            print(str(ip_src) + " -> " + str(ip_dst) +
                            " : " + "(" + str(query)[2:-1] + ")" + " : " + prediction)
                    except Exception:
                        logger.exception("Error in query_sniff when trying"
                                            "to handle other encodings")
                except Exception:
                    logger.exception("Error in query_sniff when trying"
                                        "to decode query")
    return query_sniff


def generate_menu():
    try:
        interface = input("\n[*] Enter Desired Interface "
            "(or press enter for default 'wlan0'):")
        if interface == '':
            interface = 'wlan0'
            print("[*] Using Default Interface: " + interface)
        else:
            print("[*] Using Selected Interface: " + interface)

        print("\n[*] Choose a Model:\n")
        for i, mod in enumerate(AVAILABLE_MODELS):
            print("[" + str(i) + "]" + get_model_name(mod), end="")
            if mod == DEFAULT_MODEL: print(" (default)", end="")
            print()

        chosen_model = input("\n[*] Enter Desired Model Number (or press enter "
            "for default '" + get_model_name(DEFAULT_MODEL) + "'):")
        if chosen_model != '':
            if chosen_model.isdigit() and int(chosen_model) < len(AVAILABLE_MODELS):
                model = AVAILABLE_MODELS[int(chosen_model)]
                print("[*] Using Selected Model: " + get_model_name(model))
                print()
                return interface, model
            else:
                print("[*] Wrong Selection")
        model = DEFAULT_MODEL
        print("[*] Using Default Model: " + get_model_name(model))
        print()
        return interface, model
    except KeyboardInterrupt:
        print("\n[*] Shutting Down...")
        sys.exit(0)


def get_model_name(file_string):
    """
    Returns the name of a model given the path of the file.
    """
    return file_string.split('/')[-1].split('.')[0]
    

def check_queries_in_models(queries):
    """
    Given a list of queries, it returns which models from the available models
    detect those queries as attacks. Used only for debugging purposes.
    The queries should be unique.
    Returns a dictionary where the key is the query and the value
    is a list of the models that gave a positive result.
    """
    queries_in_models = {}
    loaded_models = [joblib.load(m) for m in AVAILABLE_MODELS]
    for q in queries:
        queries_in_models[q] = []
        f = (pd.DataFrame(FV_FUNCTION(q, -1),index=[0]).fillna(0).drop('attack', 1)) # Refactor to function
        for m, mname in zip(loaded_models, AVAILABLE_MODELS):
            if m.predict(f) == [1]: queries_in_models[q].append(get_model_name(mname))
    return queries_in_models


def print_queries_in_models(queries_in_models):
    """
    Helper function to print the result of 'check_queries_in_models' in a nice format
    """
    print_result = ''
    for query, models in queries_in_models.items():
        print_result += f'{color.BOLD}{query}{color.END}\n'
        for m in models:
            print_result += f'{m}\n'
        print_result += '\n'
    return print_result



def main():
    initial_setup()
    interface, model = generate_menu()
    try:
        sniff(iface=interface, filter='port 53',
            prn=query_sniff_wrapper(joblib.load(model)), store=0)
    except OSError:
        logger.error(f"{interface} is not a valid interface!")

    print("\n[*] Shutting Down...")


if __name__ == '__main__':
    main()
