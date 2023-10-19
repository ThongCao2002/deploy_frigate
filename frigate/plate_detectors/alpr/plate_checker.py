import re

# import editdistance


class Plate_Checker:
    def __init__(self):
        """
        Define regex for Vietnam license plate number, having level 1,2,3 which is:
        level 1: recognized LP number need to match with full format of VietNam LP number.
        level 2: recognized LP number can be matched even having wrong format from 1-2 letters.
        level 3: in progress of thinking.
        """
        self.patterns_lv1 = [
            r"[\d]{2}[A-Z].[0-9]{3,5}",  # bien dan dung, xe may dien
            r"[\d]{5}[A-Z]{2}[0-9]{2}",  # bien ngoai giao
            r"[A-Z]{2}[\d]{4}",  # bien quan su
        ]
        self.patterns_lv2 = {
            r"",  # Cho phep sai 1 so o cuoi
        }
        self.patterns_lv3 = {}
        # self.spam_list = self.get_spam_list()
        self.starting_letter = [
            "TM",
            "TC",
            "TH",
            "TT",
            "TK",
            "TN",
            "KA",
            "KB",
            "KC",
            "KD",
            "KV",
            "KP",
            "KK",
            "KT",
            "AA",
            "AB",
            "AC",
            "AD",
            "QA",
            "QH",
            "QB",
            "QC",
            "QM",
            "BL",
            "BB",
            "BC",
            "BK",
            "BP",
            "BH",
            "BT",
            "HA",
            "HB",
            "HC",
            "HE",
            "HD",
            "HH",
            "HT",
            "HQ",
            "HN",
            "PA",
            "PG",
            "PK",
            "PQ",
            "PM",
            "PX",
            "PP",
            "AV",
            "AT",
            "AN",
            "AX",
            "AM",
            "VT",
            "CA",
            "CB",
            "CD",
            "CH",
            "CK",
            "CM",
            "CN",
            "CP",
            "CT",
            "CV",
        ]

    def run(self, input):
        """
        Run plate format checker
        :param: input: recognized licence plate number
        :return: format index, valid
                format index: 0 - in valid, 1 - bien dan dung, 2 - bien ngoai giao, 3 - bien quan su
        """
        for index, pattern in enumerate(self.patterns_lv1):
            # print("\n " + pattern)
            result = re.match(pattern, input)
            if not self.check_spam(input):
                return 0, False
            if result and self.check_length(result[0], input):
                # print("Tim kiem thanh cong.")
                return index + 1, True
            else:
                # print("Tim kiem khong thanh cong.")
                pass
        # print("Wrong Format: ", input)
        return 0, False

    @staticmethod
    def get_spam_list():
        with open("spam.txt") as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        spam_list = [x.strip() for x in content]
        return spam_list

    @staticmethod
    def check_length(match_string, inputs, threshold=2):
        """
        Check if recognized plate number having more characters than license plate format
        :param match_string: string matching from regex
        :param inputs: recognized license plate number
        """
        if (len(inputs) - len(match_string)) > threshold:
            return False
        return True

    def check_spam(self, input):
        """
        We create some heuristic to filter the false positive license plate
        :param input: the license plate ocr
        :return: False if false positive
        """
        if input is None or input == "":
            return False
        if len(input) > 2:
            if input[:2].isalpha() and not input[-1].isalpha():
                # Filter ocr result by some of starting-letter
                for item in self.starting_letter:
                    if input.startswith(item):
                        return True
                print("Get False Starting Letter: ", input)
                return False
        # Filter by a list of spam license plate
        # for item in self.spam_list:
        #     if (editdistance.eval(input, item)) < 3 or input[-1].isalpha():
        #         print("Get Spam: ", input)
        #         return False

        return True


if __name__ == "__main__":
    inputs = "thold8"
    checker = Plate_Checker()
    print(checker.run(inputs))
    print(checker.run(inputs))
