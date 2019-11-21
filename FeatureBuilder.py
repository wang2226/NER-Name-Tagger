# CIS 410/510: NLP - Assignment 3
# Due Nov 16th, 2019
# Haoran Wang (hwang8@cs.uoregon.edu)

import itertools
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.corpus import names
from geotext import GeoText

class FeatureBuilder:

    def __init__(self, file_path):
        # the file_path to read in
        self.file_path = file_path

        # return a list of features
        self.feature_list = list()

    def get_sentences(self):
        # split data by "\n"
        split = list()

        # read in lines
        for line in open(self.file_path, "r").readlines():
            # we have a sentence, strip "\n"
            if line == "\n":
                split.append(line)
            # separate words in a sentence, strip "\t"
            else:
                split.append(line.split("\t"))
        # print(split)

        # Use groupby function from itertools library
        groups = list()
        keys = []
        for k, g in itertools.groupby(split, lambda x: x == "\n"):
            groups.append(list(g))
            keys.append(k)

        # list of list, each list = [['word', 'POS', 'chunk', 'name tag'], ['word', 'POS', 'chunk', 'name tag']]
        sentences = list()
        for g in groups:
            if g not in [["\n"], ["\n", "\n"]]:
                sentences.append(list(g))

        return sentences

    # take in feature vector and transform it dictionary form
    def feature_vector_to_feature_dict(self, feature_vector):
        # print({"token": feature_vector[0], "pos": feature_vector[1], "chunk": feature_vector[2].strip()})
        return {"token": feature_vector[0],
                "pos": feature_vector[1],
                "chunk": feature_vector[2].strip()}

    # get the tag for each word from training file
    def get_tags(self, sentences):
        tags = list()
        for sentence in sentences:
            for feature_vector in sentence:
                tags.append(feature_vector[3].rstrip("\n"))
        return tags

    # get baseline features
    # {token, pos, chunk, token index, START, END}
    def get_baseline_features(self, sentences):

        for sentence in sentences:
            token_index = 0

            START = False
            END = False

            prev_token = ""
            next_token = ""

            for feature_vector in sentence:
                # print(training.feature_vector_to_feature_dict(feature_vector))

                # feature_list.append(training.feature_vector_to_feature_dict(feature_vector))
                feature = training.feature_vector_to_feature_dict(feature_vector)

                # add token_index to dictionary
                token_index = token_index + 1
                feature.update({"token_index": token_index})

                # Add start and end symbol
                if token_index == 1:
                    START = True
                    prev_token = "None"
                    prev_tag = "None"
                else:
                    START = False
                    prev_token = sentence[token_index - 2][0]
                    prev_tag = "@@"

                feature.update({"START": START})
                feature.update({"prev_token": prev_token})
                feature.update({"prev_tag": prev_tag})

                if token_index == len(sentence):
                    END = True
                    next_token = "None"
                else:
                    END = False
                    next_token = sentence[token_index][0]
                feature.update({"END": END})
                feature.update({"next_token": next_token})

                # add to list
                self.feature_list.append(feature)

    def add_elaborate_features(self):
        last_name = list()
        fp_last_name = open("./dist.all.last.txt", "r")
        for i in range(5000):
            last_name.append(fp_last_name.readline().split()[0])

        largest_city = list()
        for line in open("./LargestCity.txt", "r").readlines():
            largest_city.append(line.strip())

        for i in range(len(self.feature_list)):
            # self.feature_list[i]["nltk_stopword"] = self.feature_list[i]["token"] in stopwords.words("english")
            # self.feature_list[i]["is_nltk_name"] = self.feature_list[i]["token"].lower() in (n.lower() for n in names.words())
            # self.feature_list[i]["is_geo_place"] = bool( GeoText(self.feature_list[i]["token"]).cities or GeoText(self.feature_list[i]["token"]).countries)
            self.feature_list[i]["case"] = "lower" if self.feature_list[i]["token"] == self.feature_list[i]["token"].lower() else "upper"
            self.feature_list[i]["last_char"] = self.feature_list[i]["token"][-1]
            # self.feature_list[i]["is_last_name"] = bool(self.feature_list[i] in last_name)
            # self.feature_list[i]["is_city"] = bool(self.feature_list[i] in largest_city)

    def write_feature_enhanced(self, path, is_training, label):
        fout = open(path, "w")

        for i in range(len(self.feature_list)):
            out_line = self.feature_list[i].get("token", "") + "\t"

            order_dict = OrderedDict(self.feature_list[i])

            for key, value in order_dict.items():

                if key != "token":
                    out_line += str(key) + "=" + str(value) + "\t"

            if is_training is True:
                out_line += str(label[i]) + "\n"
            else:
                out_line += "\n"

            # print a "\n" at the end a sentence
            if self.feature_list[i - 1].get("END", "") is True and i != 0:
                fout.write("\n")
            fout.write(out_line)

        # print a "\n" at the very end, just because dev.name does so O.O
        fout.write("\n")

        fout.close()


if __name__ == "__main__":
    train_path = "./NAME_CORPUS_FOR_STUDENTS/train.pos-chunk-name"
    dev_path = "./NAME_CORPUS_FOR_STUDENTS/dev.pos-chunk"
    test_path = "./NAME_CORPUS_FOR_STUDENTS/test.pos-chunk"

    #####################################
    # generate features for training file
    training = FeatureBuilder(train_path)
    sentences = training.get_sentences()

    # list of features generated
    # training.feature_list = training.get_baseline_features(sentences)
    training.get_baseline_features(sentences)
    training.add_elaborate_features()

    # list of tags
    tags = training.get_tags(sentences)

    # write to file
    training.write_feature_enhanced("./feature-enhanced-training", True, tags)

    ###################################
    # generate features for development
    dev = FeatureBuilder(dev_path)
    sentences = dev.get_sentences()

    # list of features generated
    # dev.feature_list = dev.get_baseline_features(sentences)
    dev.get_baseline_features(sentences)
    dev.add_elaborate_features()

    # write to file
    dev.write_feature_enhanced("./feature-enhanced-dev", False, "")

    ###################################
    # generate features for test
    test = FeatureBuilder(test_path)
    sentences = test.get_sentences()

    # list of features generated
    # test.feature_list = test.get_baseline_features(sentences)
    test.get_baseline_features(sentences)
    test.add_elaborate_features()

    # write to file
    test.write_feature_enhanced("./feature-enhanced-test", False, "")
