import os
from plate_checker import Plate_Checker


def get_file_folder(root):
    out = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-4:] == '.jpg':
                out.append(os.path.join(path, name))
    return out


if __name__ == "__main__":
    checker = Plate_Checker()
    check_wrong = []
    files = get_file_folder("/home/can/AI_Camera/Dataset/License_Plate/ocr_20k/train/10K_15_10")
    for file in files:
        label = os.path.basename(file).split("_")[0]
        if not checker.run(label):
            check_wrong.append(label)
    for text in check_wrong:
        print(text)
    print(len(check_wrong))
