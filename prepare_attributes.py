import multiprocessing

import nltk
from nltk.tokenize import RegexpTokenizer

from nltk.tag import StanfordPOSTagger, PerceptronTagger
from miscc.config import cfg
from tqdm import tqdm
import argparse
import torch.utils.data as data
import os
import sys
import numpy.random as random

from miscc.utils import mkdir_p

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from datasets import TextDataset

"The two packages are from stanford-postagger-full-2015-04-20.zip"
# taggar_file_path = \
#     '../nltk_data/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger'
# path_to_jar = \
#     '../nltk_data/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a EE-GAN network')
    parser.add_argument('--taggar_mode', default='standford', type=str)
    parser.add_argument('--taggar_file_path',
    default='../nltk_data/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger', type=str)
    parser.add_argument('--jar_file_path',
    default='../nltk_data/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar', type=str)
    parser.add_argument('--data_dir', default='../data/coco2014', type=str)
    parser.add_argument('--cap_filename', default='caption.pickle', type=str)
    parser.add_argument('--attr_filename', default='EE-GAN.pickle', type=str)
    parser.add_argument('--dataset_name', default='coco', type=str)
    args = parser.parse_args()
    return args

class PrepareAttrs:

    def __init__(self, args):

        self.dataset_name = args.dataset_name
        self.embeddings_num = 5 if self.dataset_name == 'coco' else 10
        # self.tokenizer, self.taggar, self.chunk_parsers, self.split_chunk_parsers = \
        self.parser_func = self.load_attr_parser(self.dataset_name, args.taggar_file_path, args.jar_file_path,
                                                 args.taggar_mode)

        self.train_captions, self.test_captions, self.train_names, self.test_names, self.wordtoix, self.ixtoword = \
            self.load_text_embedding_info(args.data_dir, args.cap_filename)

    @staticmethod
    def load_text_embedding_info(data_dir, caps_name):
        filepath = os.path.join(data_dir, caps_name)
        train_names = TextDataset.load_filenames(data_dir, 'train')
        test_names = TextDataset.load_filenames(data_dir, 'test')
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
        return train_captions, test_captions, train_names, test_names, wordtoix, ixtoword

    @staticmethod
    def load_attr_parser(dataset_name, taggar_file_path, jar_file_path, taggar_mode='stanford'):

        tokenizer = RegexpTokenizer(r'\w+')

        if taggar_mode == 'stanford':
            taggar = StanfordPOSTagger(taggar_file_path, jar_file_path)
            print("using stanford nlp taggar")
        else:
            # a simple implement to achieve taggar
            taggar = PerceptronTagger()

        if dataset_name == 'bird':
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_cub_parser()
        elif dataset_name == 'flower':
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_oxford_parser()
        else:
            chunk_parsers, split_chunk_parsers = PrepareAttrs.define_coco_parser()

        print("conducting %s dataset" % dataset_name)
        return [tokenizer, taggar, chunk_parsers, split_chunk_parsers]

    @staticmethod
    def define_cub_parser():
        # Rules defined by DAE-GAN
        # "NP: {<DT>*<JJ>*<CC|IN>*<JJ>+<NN|NNS>+|<DT>*<NN|NNS>+<VBZ>+<JJ>+<IN|CC>*<JJ>*}"

        # The following rules are defined by us
        ADJ = "<JJ.*|VBD|VBN|VBG>"
        DET = "<PDT|DT|PRP.*|POS>"
        PRON = "<IN|WP*|WDT>"
        V = "<VB|VBP|VBZ>"

        # black wings and beak
        grammar_1 = "AND: {<CC>%s?<NN.*>+}\n" \
                    "NP: {%s?%s+<NN.*>+<AND>*}" % (DET, DET, ADJ)

        # black and white wings
        grammar_2 = "AND2: {<CC>%s+}\n" \
                    "NP: {%s?%s+<AND2>*<NN.*>+}" % (ADJ, DET, ADJ)

        # wings (that) is brown and green
        # orange is on the wings
        # bird has some red on the black wings

        # N do/be N/ADJ
        grammar_3 = "INTRO: {<NN.*>+%s?%s<IN>?}\n" \
                    "AND2: {<CC>%s+}\n" \
                    "NP: {<INTRO>%s?%s*<AND2>*<NN.*>*}" % (PRON, V, ADJ, DET, ADJ)

        # patch on its black head
        grammar_4 = "LOC: {<IN>%s?%s*<NN.*>+}\n" \
                    "NP: {<NN.*>+<LOC>+}" % (DET, ADJ)

        # N do
        grammar_3_split = "NP: {<NN.*>+%s+%s*}" % (V, DET)

        grammars = [grammar_1, grammar_2, grammar_3, grammar_4]
        split_grammars = [grammar_3_split]

        set_of_chunk_parser = [nltk.RegexpParser(grammar) for grammar in grammars]
        split_set_of_chunk_parser = [nltk.RegexpParser(grammar) for grammar in split_grammars]

        return set_of_chunk_parser, split_set_of_chunk_parser

    @staticmethod
    def define_oxford_parser():
        ADJ = "<JJ.*|VBD|VBN|VBG>"
        DET = "<PDT|DT|PRP.*|POS>"
        PRON = "<IN|WP*|WDT>"
        V = "<VB|VBP|VBZ>"

        # black wings and beak
        grammar_1 = "AND: {<CC>%s?<NN.*>+}\n" \
                    "NP: {%s?%s+<NN.*>+<AND>*}" % (DET, DET, ADJ)

        # black and white wings
        grammar_2 = "AND2: {<CC>%s+}\n" \
                    "NP: {%s?%s+<AND2>*<NN.*>+}" % (ADJ, DET, ADJ)

        # wings (that) is brown and green
        # orange is on the wings
        # bird has some red on the black wings

        # N do/be N/ADJ
        grammar_3 = "INTRO: {<NN.*>+%s?%s<IN>?}\n" \
                    "AND2: {<CC>%s+}\n" \
                    "NP: {<INTRO>%s?%s*<AND2>*<NN.*>*}" % (PRON, V, ADJ, DET, ADJ)

        # patch on its black head
        grammar_4 = "AND: {<CC>%s?<NN.*>+}\n" \
                    "LOC: {<IN>%s?%s*<NN.*>+<AND>*}\n" \
                    "NP: {<NN.*>+<LOC>+}" % (DET, DET, ADJ)

        # N do
        grammar_3_split = "NP: {<NN.*>+%s+%s*}" % (V, DET)

        grammars = [grammar_1, grammar_2, grammar_3, grammar_4]
        split_grammars = [grammar_3_split]

        set_of_chunk_parser = [nltk.RegexpParser(grammar) for grammar in grammars]
        split_set_of_chunk_parser = [nltk.RegexpParser(grammar) for grammar in split_grammars]

        return set_of_chunk_parser, split_set_of_chunk_parser

    @staticmethod
    def define_coco_parser():

        # the rules are defined in DAE-GAN
        # grammar = "NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|" \
        #           "<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|" \
        #           "<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|" \
        #           "<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}"

        ADJ = "<JJ.*|VBD|VBN|VBG|RB>"
        DET = "<PDT|DT|PRP.*|CD>"
        PRON = "<IN|WP*|WDT>"
        V = "<VB|VBD|VBG|VBN|VBP|VBZ>"

        # a set of books
        # reflection of a dog
        grammar_1 = "RES: {%s%s?<NN.*>+}\n" \
                    "NP: {%s*%s*<NN.*>+<RES>?}" % (PRON, DET, DET, ADJ)

        # do some thing
        grammar_2 = "NP: {<NN.*>+%s+%s*%s*%s*<NN.*>*}" % (V, PRON, DET, ADJ)

        # table in a small room
        # christmas trees in front of a building
        grammar_3 = "LOC: {%s%s*%s*<NN.*>+%s*%s*%s*<NN.*>*}\n" \
                    "NP: {<NN.*>+<LOC>}" % (PRON, DET, ADJ, PRON, DET, ADJ)

        grammars = [grammar_1, grammar_2, grammar_3]

        set_of_chunk_parser = [nltk.RegexpParser(grammar) for grammar in grammars]
        split_set_of_chunk_parser = []

        return set_of_chunk_parser, split_set_of_chunk_parser

    """
    given filenames, do_process_attrs processes the captions by calling do_parse_one_caption;
    main conduct the core task by using multi_thread_processing; 
    main_test could do a simple test for a few data.
    """
    @staticmethod
    def do_parse_one_caption(parser_func, cap):
        """
        tokenizer, chunk_parsers and split_chunk_parsers
        """
        [tokenizer, taggar, chunk_parsers, split_chunk_parsers] = parser_func
        if isinstance(cap, str):
            tokens = tokenizer.tokenize(cap.lower())
        else:
            tokens = cap

        tags = taggar.tag(tokens)
        attr_set = set()

        for chunk_parser in chunk_parsers:
            tree = chunk_parser.parse(tags)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                myPhrase = []
                for item in subtree.leaves():
                    myPhrase.append(item[0])
                tmp = " ".join(myPhrase)
                attr_set.add(tmp)

        if split_chunk_parsers is not None:
            for chunk_parser in split_chunk_parsers:
                tree = chunk_parser.parse(tags)
                for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                    myPhrase = []
                    for item in subtree.leaves():
                        myPhrase.append(item[0])
                    tmp = " ".join(myPhrase)
                    attr_set.discard(tmp)

        revs = []
        for attr_str in attr_set:
            tmp = attr_str.split(" ")
            revs.append(tmp)

        return revs

    def process_attrs_from_pickle(self, cap_tokens):

        wordtoix = self.wordtoix
        ixtoword = self.ixtoword
        all_attr_tokens = []
        for i in tqdm(range(len(cap_tokens))):
            cap = [ixtoword[token_ix] for token_ix in cap_tokens[i]]
            attrs = self.do_parse_one_caption(self.parser_func, cap)
            attrs_tokens = list()

            for attr in attrs:
                tokens = list()
                for w in attr:
                    if w in wordtoix:
                        tokens.append(wordtoix[w])
                attrs_tokens.append(tokens)

            all_attr_tokens.append(attrs_tokens)  # [cnt, attrs_num, num_words]

        print("one batch is finished")
        # the attr tokens are returned
        return all_attr_tokens

    @staticmethod
    def multi_thread_processing(process_func, process_data, one_batch_nums, using_works,
                                text_data_dir=None):

        f_len = len(process_data)
        n_batch = f_len // one_batch_nums

        last_st = n_batch * one_batch_nums
        last_batch = process_data[last_st:]
        if len(last_batch) > 0:
            res = 1
        else:
            res = 0

        print("batch num is %d " % (n_batch + res))
        pool = multiprocessing.Pool(processes=using_works)
        results = []

        for i in range(n_batch):
            st = i * one_batch_nums
            ed = st + one_batch_nums
            batch = process_data[st:ed]
            results.append(pool.apply_async(process_func, args=(batch, text_data_dir, )))

        if res == 1:
            results.append(pool.apply_async(process_func, args=(last_batch, text_data_dir, )))

        pool.close()
        pool.join()

        merge_rev = []
        for res in results:
            merge_rev += res.get()

        print(len(merge_rev))
        return merge_rev

    def main(self, save_pickle_path, one_batch_nums=50, using_works=16):

        if os.path.exists(save_pickle_path):
            print("%s already exists." % save_pickle_path)
            return
        else:
            func = self.process_attrs_from_pickle
            data = self.train_captions
            train_attrs = self.multi_thread_processing(func, data, one_batch_nums, using_works)
            data = self.test_captions
            test_attrs = self.multi_thread_processing(func, data, one_batch_nums, using_works)

        with open(save_pickle_path, 'wb') as f:
            pickle.dump([train_attrs, test_attrs], f, protocol=2)
            print('Save to: ', save_pickle_path)

    def sampling(self, sampling_num=10):

        wordtoix, ixtoword = self.wordtoix, self.ixtoword
        train_captions, test_captions = self.train_captions, self.test_captions
        train_captions, test_captions = train_captions[:sampling_num], test_captions[:sampling_num]
        train_attrs_tokens = self.process_attrs_from_pickle(train_captions)
        test_attrs_tokens = self.process_attrs_from_pickle(test_captions)

        train_caps = [cap_token_to_str(cap_token, ixtoword) for cap_token in train_captions]
        test_caps = [cap_token_to_str(cap_token, ixtoword) for cap_token in test_captions]

        train_attrs = [attrs_token_to_str(attr_token, ixtoword) for attr_token in train_attrs_tokens]
        test_attrs = [attrs_token_to_str(attr_token, ixtoword) for attr_token in test_attrs_tokens]

        caps = train_caps + test_caps
        attrs = train_attrs + test_attrs

        for ix in range(len(caps)):
            print(caps[ix])
            print(str(attrs) + "\n")


def cap_token_to_str(cap_token, ixtoword):
    """
    return str
    """
    rev_caps = [ixtoword[token_ix] for token_ix in cap_token]
    rev_caps = " ".join(rev_caps)
    return rev_caps

def attrs_token_to_str(attrs_token_list, ixtoword):
    """
    return a list of str
    """
    rev_attrs = []
    for i in range(len(attrs_token_list)):
        one_attr = attrs_token_list[i]
        list_str_one_attr = [ixtoword[token_ix] for token_ix in one_attr]
        rev_attrs.append(" ".join(list_str_one_attr))
    return rev_attrs


if __name__ == '__main__':

    args = parse_args()
    attr_pickle_dir = os.path.join(args.data_dir, "attributes")
    mkdir_p(attr_pickle_dir, rm_exist=False)
    attr_pickle_path = os.path.join(attr_pickle_dir, args.attr_filename)

    pre = PrepareAttrs(args)
    pre.main(attr_pickle_path)

