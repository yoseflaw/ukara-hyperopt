from typing import List
import string


def get_vocab(fpath: str, fout: str) -> List:
    unique_token = []
    with open(fpath, "r") as fin:
        for content in fin.readlines():
            content = content.translate(str.maketrans('', ' ', string.punctuation))
            content = content.replace("\n","")
            content = content.lower()
            tokens = content.split(" ")
            for token in tokens:
                strip_token = token.strip() + "\n"
                if strip_token not in unique_token:
                    unique_token.append(strip_token)
    with open(fout, "w") as fout:
        fout.writelines(sorted(unique_token))
    return unique_token


def main():
    tokens_a = get_vocab("../lib/response_only_A.txt", "../lib/vocab_A.txt")
    tokens_b = get_vocab("../lib/response_only_B.txt", "../lib/vocab_B.txt")

if __name__=="__main__":
    main()