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
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from datasets import TextDataset

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

    parser.add_argument('--gpu', dest='gpu_ids', type=str, default="0")
    parser.add_argument('--output_dir', dest='output_dir',
                        help='the path to save models and images',
                        default='EE-GAN bird', type=str)
    parser.add_argument('--debug_output_dir', dest='debug_output_dir',
                        help='the path to save models and images in debug mode',
                        default='Debug', type=str)
    parser.add_argument('--debug', action="store_true", help='using debug mode')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=3407)
    args = parser.parse_args()
    return args

class PrepareAttrs:

    def __init__(self, data_dir, cap_name, dataset_name='bird',
                 embeddings_num=cfg.TEXT.CAPTIONS_PER_IMAGE, taggar_mode='stanford'):

        self.embeddings_num = embeddings_num
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.chunk_parsers, self.split_chunk_parsers = self.define_parser(dataset_name)
        self.taggar = self.load_taggar(taggar_mode)

        self.train_captions, self.test_captions, self.train_names, self.test_names, self.wordtoix, self.ixtoword = \
            self.load_text_embedding_info(self.data_dir, cap_name)

        print("finish init")

    """
    the following six functions are used to load the prepare the caption file, taggar and parser 
    """
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
    def load_taggar(taggar_mode='stanford'):
        if taggar_mode == 'stanford':
            taggar_file_path = \
                '../nltk_data/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger'
            path_to_jar = \
                '../nltk_data/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar'

            rev_taggar = StanfordPOSTagger(model_filename, path_to_jar)

            print("using stanford nlp taggar")
        else:
            # more simple
            rev_taggar = PerceptronTagger()

        return rev_taggar

    def define_parser(self, dataset_name):
        if dataset_name == 'CUB':
            chunk_parsers, split_chunk_parsers = self.define_cub_parser()
        elif dataset_name == 'Oxford':
            chunk_parsers, split_chunk_parsers = self.define_oxford_parser()
        else:
            chunk_parsers, split_chunk_parsers = self.define_coco_parser()
        print("conducting %s dataset" % dataset_name)

        return chunk_parsers, split_chunk_parsers

    @staticmethod
    def define_cub_parser():
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
    def do_parse_one_caption(self, cap):
        """
        tokenizer, chunk_parsers and split_chunk_parsers
        """
        if isinstance(cap, str):
            tokens = self.tokenizer.tokenize(cap.lower())
        else:
            tokens = cap

        tags = self.taggar.tag(tokens)
        attr_set = set()

        for chunk_parser in self.chunk_parsers:
            tree = chunk_parser.parse(tags)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                myPhrase = []
                for item in subtree.leaves():
                    myPhrase.append(item[0])
                tmp = " ".join(myPhrase)
                attr_set.add(tmp)

        if self.split_chunk_parsers is not None:
            for chunk_parser in self.split_chunk_parsers:
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

    def process_attrs_from_row(self, filenames, text_data_dir):
        wordtoix = self.wordtoix
        all_attr_tokens = []
        # all_caps = []
        all_caps = []
        for i in tqdm(range(len(filenames))):
            cap_path = os.path.join(text_data_dir, "%s.txt" % filenames[i])
            with open(cap_path, "r") as f:
                #captions = f.read().decode('utf8').split('\n')
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue

                    cap = cap.replace("\ufffd\ufffd", " ")
                    attrs = self.do_parse_one_caption(cap)
                    attrs_tokens = list()

                    for attr in attrs:
                        tokens = list()
                        for w in attr:
                            if w in wordtoix:
                                tokens.append(wordtoix[w])
                        attrs_tokens.append(tokens)

                    all_attr_tokens.append(attrs_tokens)  # [cnt, attrs_num, num_words]
                    all_caps.append(cap)

                    cnt += 1
                    if cnt == self.embeddings_num:
                        break

                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d, len(captions)=%d, embedding_num=%d'
                          % (filenames[i], cnt, len(captions), self.embeddings_num))

        print("one batch is finished")
        # the attr tokens are returned
        return all_attr_tokens, all_caps

    def process_attrs_from_pickle(self, cap_tokens, text_data_dir=None):

        wordtoix = self.wordtoix
        ixtoword = self.ixtoword
        all_attr_tokens = []
        for i in tqdm(range(len(cap_tokens))):
            cap = [ixtoword[token_ix] for token_ix in cap_tokens[i]]
            attrs = self.do_parse_one_caption(cap)
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
        # [0, 100], [100, 200]
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

    def main_test(self, sampling_num=10, load_row=False):

        wordtoix, ixtoword = self.wordtoix, self.ixtoword

        # method 1.
        if load_row:
            train_names, test_names = self.train_names, self.test_names
            train_names, test_names = train_names[:sampling_num], test_names[:sampling_num]
            text_data_dir = os.path.join(data_dir, 'text/train')
            train_attrs_tokens, train_caps = self.process_attrs_from_row(text_data_dir, train_names)
            text_data_dir = os.path.join(data_dir, 'text/test')
            test_attrs_tokens, test_caps = self.process_attrs_from_row(text_data_dir, test_names)

        else:
            # method 2.
            train_captions, test_captions = self.train_captions, self.test_captions
            train_captions, test_captions = train_captions[:sampling_num], test_captions[:sampling_num]
            train_attrs_tokens = self.process_attrs_from_pickle(train_captions)
            test_attrs_tokens = self.process_attrs_from_pickle(test_captions)

            train_caps = [cap_token_to_str(cap_token, ixtoword) for cap_token in train_captions]
            test_caps = [cap_token_to_str(cap_token, ixtoword) for cap_token in test_captions]

        attrs_tokens = train_attrs_tokens + test_attrs_tokens
        caps = train_caps + test_caps

        for i in range(len(caps)):
            one_attrs_token = attrs_tokens[i]
            attrs_str = []
            for j in range(len(one_attrs_token)):
                one_attr = [ixtoword[token_ix] for token_ix in one_attrs_token[j]]
                one_attr_str = " ".join(one_attr)
                attrs_str.append(one_attr_str)

            print(caps[i])
            print(str(attrs_str) + "\n")

    def main(self, save_attr_name, one_batch_nums=50, using_works=16, load_row=False):
        filepath = os.path.join(self.data_dir, save_attr_name)
        if os.path.exists(filepath):
            print("%s already exists." % filepath)
            return

        if load_row:
            # print("start training extraction")
            func = self.process_attrs_from_row
            data = self.train_names
            text_data_dir = os.path.join(data_dir, 'text/train')
            train_attrs = \
                self.multi_thread_processing(func, data, one_batch_nums, using_works, text_data_dir)

            data = self.test_names
            text_data_dir = os.path.join(data_dir, 'text/test')
            test_attrs = \
                self.multi_thread_processing(func, data, one_batch_nums, using_works, text_data_dir)

        else:
            func = self.process_attrs_from_pickle
            data = self.train_captions
            train_attrs = self.multi_thread_processing(func, data, one_batch_nums, using_works)
            data = self.test_captions
            test_attrs = self.multi_thread_processing(func, data, one_batch_nums, using_works)

        with open(filepath, 'wb') as f:
            pickle.dump([train_attrs, test_attrs], f, protocol=2)
            print('Save to: ', filepath)

def no_len_data_ix():
    train_ixs = [52503, 54325, 70930, 76547, 78283, 78502, 86441, 94664, 102765, 110352, 127808, 135576, 137440,
                 146169, 147096, 147391, 148767, 154925, 156753, 170442, 186145, 192562, 201246, 203104, 238464, 275231,
                 277205, 278591, 279250, 294232, 299951, 305341, 307028, 327477, 330317, 332078, 347144, 359328, 364475,
                 390906, 393688, 398283, 412869]
    test_ixs = [66, 5834, 16774, 25594, 30114, 31140, 44696, 52989, 60244, 67394, 81387, 82228, 83586, 83876, 87074,
                93227, 96036, 112097, 112627, 116180, 118519, 121051, 130011, 139478, 152134, 154766, 168233, 171882,
                173423, 190568, 196893, 197773, 202343]

    return train_ixs, test_ixs

def sampling(data_dir, cap_name, attr_name, sampling_num):

    caps_filepath = os.path.join(data_dir, cap_name)
    attrs_filepath = os.path.join(data_dir, attr_name)

    with open(attrs_filepath, 'rb') as f:
        x = pickle.load(f)
        train_attrs, test_attrs = x[0], x[1]
        del x
        print('Load from: ', attrs_filepath)

    with open(caps_filepath, 'rb') as f:
        x = pickle.load(f)
        train_caps, test_caps = x[0], x[1]
        ixtoword, wordtoix = x[2], x[3]
        del x
        # n_words = len(ixtoword)
        print('Load from: ', caps_filepath)

    print("training")
    # train_rev_ixs = check(train_caps, train_attrs, ixtoword)

    # print("testing")
    # test_rev_ixs = check(test_caps, test_attrs, ixtoword)
    #
    # rev_ixs = train_rev_ixs + test_rev_ixs
    # print(train_rev_ixs)
    #
    # print(test_rev_ixs)

    # len(train_caps) = 413915
    # len(train_attrs) = 414154

    len_train, len_test = len(train_caps), len(test_caps)
    rev_train_caps = []
    rev_train_attrs = []
    rev_test_caps = []
    rev_test_attrs = []

    for _ in range(sampling_num):
        cap, attr = sampling_one(train_caps, train_attrs, len_train, ixtoword)
        rev_train_caps.append(cap)
        rev_train_attrs.append(attr)

    for _ in range(sampling_num):
        cap, attr = sampling_one(test_caps, test_attrs, len_test, ixtoword)
        rev_test_caps.append(cap)
        rev_test_attrs.append(attr)

    # rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs

    display(rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs)

def sampling_one(caps_token, attrs_token, len_set, ixtoword):
    sample_ix = random.randint(0, len_set)
    cap_token = caps_token[sample_ix]
    attrs_token_set = attrs_token[sample_ix]
    rev_caps = [ixtoword[token_ix] for token_ix in cap_token]
    rev_caps = " ".join(rev_caps)
    rev_attrs = []

    for i in range(len(attrs_token_set)):
        one_attr = attrs_token_set[i]
        list_str_one_attr = [ixtoword[token_ix] for token_ix in one_attr]
        rev_attrs.append(" ".join(list_str_one_attr))

    return rev_caps, rev_attrs

def display(rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs):

    n = len(rev_train_caps)
    print("training samples:")
    for i in range(n):
        print(rev_train_caps[i])
        print(rev_train_attrs[i])

    print("\ntesting samples")
    for i in range(n):
        print(rev_test_caps[i])
        print(rev_test_attrs[i])

def check(caps_token, attrs_token, ixtoword):

    rev_ixs = []
    for ix, cap in enumerate(caps_token):
        if len(cap) == 0 or len(attrs_token[ix]) == 0:
            cap_str = cap_token_to_str(cap, ixtoword)
            attr_str_list = attrs_token_to_str(attrs_token[ix], ixtoword)
            print("ix: %d" % ix)
            print(cap_str)
            print(attr_str_list)
            print("\n")
            rev_ixs.append(ix)

    return rev_ixs

"""
there are some questions in this function
"""
def check2(Pre_Handle: PrepareAttrs, data_dir, split, cap_ixs):
    """
    filenames support read text data from row;
    cap_tokens support read from pickle;
    ixs denote the caption idx
    """

    text_data_dir = os.path.join(data_dir, "text", split)
    if split == 'train':
        filenames = Pre_Handle.train_names
        cap_tokens = Pre_Handle.train_captions
    else:
        filenames = Pre_Handle.test_names
        cap_tokens = Pre_Handle.test_captions

    img_ixs = [ix // 5 for ix in cap_ixs]
    img_ixs_bias = [ix % 5 for ix in cap_ixs]

    # from row text data
    input_filenames = [filenames[ix] for ix in img_ixs]
    _, caption_list = Pre_Handle.process_attrs_from_row(text_data_dir, input_filenames)

    # from pickle
    ixtoword = Pre_Handle.ixtoword
    embedding_emb = Pre_Handle.embeddings_num

    for ix in range(len(cap_ixs)):
        ix_bias = img_ixs_bias[ix]
        print(caption_list[ix * embedding_emb + ix_bias])
        one_cap = cap_token_to_str(cap_tokens[ix], ixtoword)
        print(one_cap)
        print("\n")

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

    # the code is used to test the attributes dataset
    data_dir = '../data/coco2014/'
    dataset_name = 'coco'
    cap_name = "captions.pickle"
    attrs_name = "attributes/ours_coco.pickle"

    #train_ixs, test_ixs = no_len_data_ix()

    PreHandle = PrepareAttrs(data_dir, cap_name, dataset_name=dataset_name,
                             embeddings_num=5, taggar_mode='stanford')

    #         # 82783 * 5
    #         # 40470 * 5
    PreHandle.main(attrs_name, one_batch_nums=1000, using_works=16, load_row=False)

    # sampling(data_dir, cap_name, attrs_name, 10)

    # PreHandle.main_test(10, load_row=True)

    # check2(PreHandle, data_dir, 'train', train_ixs)

    # sampling(data_dir, cap_name, attrs_name, 10)
    # PreHandle.main(attrs_name, one_batch_nums=64, using_works=12, load_row=False)

    # PreHandle.main_test()
    # PreHandle.do_it_test(cap_name=cap_name, save_attr_name=attrs_name, sampling_num=1)

    # rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs = \
    #     PreHandle.sampling(cap_name, attrs_name, sampling_num=10)
    #
    # PreHandle.display(rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs)

    # image_transform = transforms.Compose([
    #     transforms.Resize(int(imsize * 76 / 64)),
    #     transforms.RandomCrop(imsize),
    #     transforms.RandomHorizontalFlip()])
    #
    # dataset = TextDataset(data_dir, 'train',
    #                       base_size=cfg.TREE.BASE_SIZE,
    #                       transform=image_transform,
    #                       get_mismatch_pair=False)
    #
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, drop_last=True,
    #     shuffle=True, num_workers=0)
    #
    # imgs, caps, cap_len, attrs, attrs_num, attrs_len, cls_id, key = next(iter(dataloader))
    # print("over loading")

    # prepare_attrs = PrepareAttrs(data_dir, dataset_name='CUB', embeddings_num=10)
    # attrs_name = "attributes/DAE.pickle"
    # caps_name = "attn_captions.pickle"

    # prepare_attrs.DAE_do_it(attrs_name)
    # caps_filepath = os.path.join(data_dir, caps_name)
    # attrs_filepath = os.path.join(data_dir, attrs_name)
    #
    # sampling_num = 5
    # rev_train_caps, rev_test_caps, rev_train_attrs, rev_test_attrs = \
    #     prepare_attrs.sampling(caps_filepath, attrs_filepath, sampling_num)
    #
    # for i in range(sampling_num):
    #     print(rev_train_caps[i])
    #     print(rev_train_attrs[i])
    #     print(rev_test_caps[i])
    #     print(rev_test_attrs[i])
