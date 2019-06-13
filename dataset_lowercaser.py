"""
Script which converts dataset with BIO markup into lowercased version

"""
import argparse
import re
from settings import TAGS_MAP

import random
def flip(p):
    return (random.random() < p)


def main(cased_dataset_path, target_dataset, lowercasing_probability=1.0):
    boi_tags = list(TAGS_MAP.keys())

    escaped_tags = [re.escape(tag) for tag in boi_tags]
    alternatives = "|".join(escaped_tags)

    line_regexp = "^(.*?)(\s+)(%s)$" % alternatives
    # "^(.*?)\s+(O|I\-PER|B\-PER|I\-ORG|B\-ORG|B\-LOC|I\-LOC)$"
    print(line_regexp)
    p = re.compile(line_regexp)

    with open(target_dataset, 'w') as target_file:

        with open(cased_dataset_path) as file:
            for line in file:
                if '<DOCSTART>' in line or not line.strip():
                    target_file.write(line)
                    continue
                else:
                    # line with some text and tag
                    matched = p.match(line)
                    if matched:
                        print("MATCH")
                        print(line)
                        print(matched.group(1))
                        print(matched.group(3))
                        if lowercasing_probability == 1.0 or flip(lowercasing_probability):
                            modified_text = matched.group(1).lower()
                        else:
                            modified_text = matched.group(1)

                        target_line = "%s%s%s\n" % (modified_text, matched.group(2), matched.group(3))
                        print("target_line")
                        print(target_line)
                        target_file.write(target_line)
                        print("MATCH")

                    else:
                        raise Exception("Can not match the line: %s" % line)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cased_dataset', help='Path to BIO dataset with Camel Cased strings',
                        default='data/train.txt', type=str)
    parser.add_argument('--target_path', help='Path to save the resulting dataset',
                        default='data/train_lowercased.txt', type=str)
    parser.add_argument('--lowercasing_probability', help='Probability to lowercase the string (0-1.0]',
                        default=1.0, type=float)
    args = parser.parse_args()
    main(args.cased_dataset, args.target_path, args.lowercasing_probability)
