import csv
import pandas as pd


class BRO_DNS_record(object):

    def __init__(self, row):
        """
        Create DNS Record from fields row
        """
        self.ts = self.fetch_field(row, 0)
        self.uid = self.fetch_field(row, 1)
        self.origin_IP = self.fetch_field(row, 2)
        self.origin_port = self.fetch_field(row, 3)
        self.dest_IP = self.fetch_field(row, 4)
        self.dest_port = self.fetch_field(row, 5)
        self.protocol = self.fetch_field(row, 6)
        self.trans_ID = self.fetch_field(row, 7)
        self.round_trip_time = self.fetch_field(row, 8)
        self.query = self.fetch_field(row, 9)
        self.qclass = self.fetch_field(row, 10)
        self.qclass_name = self.fetch_field(row, 11)
        self.query_type = self.fetch_field(row, 12)
        self.query_type_name = self.fetch_field(row, 13)
        self.response_code = self.fetch_field(row, 14)
        self.response_code_name = self.fetch_field(row, 15)
        aa = self.fetch_field(row, 16)
        self.authoritative_answer = (True if aa=='T' else
        (False if aa=='F' else aa))
        tc = self.fetch_field(row, 17)
        self.truncated_message = (True if tc=='T' else
        (False if tc=='F' else tc))
        rd = self.fetch_field(row, 18)
        self.recursion_desired = (True if rd=='T' else
        (False if rd=='F' else rd))
        ra = self.fetch_field(row, 19)
        self.recursion_available = (True if ra=='T' else
        (False if ra=='F' else ra))
        self.Z = self.fetch_field(row, 20)
        ans = self.fetch_field(row, 21)
        self.answers = ans.split(',') if ans != None else ans
        cach = self.fetch_field(row, 22)
        self.caching_intervals = cach.split(',') if cach != None else cach
        rj = self.fetch_field(row, 23)
        self.rejected = (True if rj=='T' else (False if rj=='F' else rj))
        self.total_answers = self.fetch_field(row, 24)
        self.total_replies = self.fetch_field(row, 25)
        sq = self.fetch_field(row, 26)
        self.saw_query = (True if sq=='T' else (False if sq=='F' else sq))
        sr = self.fetch_field(row, 27)
        self.saw_reply = (True if sr=='T' else (False if sr=='F' else sr))
        self.auth = self.fetch_field(row, 28)
        self.additional_responses = self.fetch_field(row, 29)

    @staticmethod
    def fetch_field(row, index):
        """
        Returns the field corresponding to the index in row if it exists
        """
        if index < len(row):
            return row[index].strip()
        else:
            return None

    @classmethod
    def parse_BRO_log_file(cls, infile):
        """
        Parses all the fields in a BRO log file
        infile: log file
        """
        dictionary_list = []
        with open(infile) as inf:
            for row in csv.reader(inf, delimiter='\t'):
                if row and row[0][0] != '#':
                    dic = vars(cls(row))
                    dictionary_list.append(dic)
            # Dictionaries don't have an order, it's possible that this won't
            # return the keys in order. A method __dict__shoud be added to
            # the class that returns an ordered dictionary
            columns = vars(cls(row)).keys()
            df = pd.DataFrame(dictionary_list, columns=columns)
        return df

    @staticmethod
    def get_not_A_records(df):
        not_A = df.loc[df['query_type_name'] != 'A']
        with open('not_A.txt', 'w') as outf:
            outf.write(not_A.to_string())
