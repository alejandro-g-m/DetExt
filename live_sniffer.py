import sys
import glob
import logging
from sklearn.externals import joblib
from scapy.all import *
from feature_vector_creation import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load models using the feature vector with entropy and ratios
directory = './models'
function_directory = '/extract_features_entropy_and_ratios'
default_model = directory + function_directory + '/decision_trees+knn.pkl'
FV_function = extract_features_entropy_and_ratios

available_models = sorted(
[mod for mod in glob(directory + function_directory + '/*.pkl')])


def query_sniff_wrapper(model):
    """
    Wrapper function in order to pass the "model" parameter to query_sniff from
    the sniff function.
    """
    def query_sniff(packet):
        """
        Analayses a sniffed packet: if it has DNS information and the DNS query
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
                        features = (pd.DataFrame(FV_function(subdomain, -1),
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
                            features = (pd.DataFrame(FV_function(subdomain, -1),
                            index=[0]).fillna(0).drop('attack', 1))
                            if model.predict(features) == 0:
                                prediction = color.GREEN + "LEGIT" + color.END
                            else:
                                prediction = color.RED + "ATTACK" + color.END
                            print(str(ip_src) + " -> " + str(ip_dst) +
                            " : " + "(" + str(query)[2:-1] + ")" + " : " + prediction)
                    except Exception as e:
                        logger.error('Exception occurred: ' + str(e))
                        logger.error(traceback.format_exc())
                except Exception as e:
                    logger.error('Exception occurred: ' + str(e))
                    logger.error(traceback.format_exc())
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
        for i, mod in enumerate(available_models):
            print("[" + str(i) + "]" + get_model_name(mod), end="")
            if mod == default_model: print(" (default)", end="")
            print()

        chosen_model = input("\n[*] Enter Desired Model Number " +
        "(or press enter for default '" + get_model_name(default_model) + "'):")
        if chosen_model != '':
            if chosen_model.isdigit() and int(chosen_model) < len(available_models):
                model = available_models[int(chosen_model)]
                print("[*] Using Selected Model: " + get_model_name(model))
                print()
                return interface, model
            else:
                print("[*] Wrong Selection")
        model = default_model
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
    Given a list of queries, it prints the models from the available models
    that detect those queries as attacks.
    Used for debugging purposes.
    """
    loaded_models = [joblib.load(m) for m in available_models]
    for q in queries:
        print(color.BOLD, q, color.END, "\n")
        f = (pd.DataFrame(FV_function(q, -1),index=[0]).fillna(0).drop('attack', 1))
        for m, mname in zip(loaded_models, available_models):
            if m.predict(f) == [1]: print(get_model_name(mname))
        print("\n")


if __name__ == '__main__':

    interface, model = generate_menu()
    sniff(iface=interface, filter='port 53',
    prn=query_sniff_wrapper(joblib.load(model)), store=0)

    print("\n[*] Shutting Down...")
