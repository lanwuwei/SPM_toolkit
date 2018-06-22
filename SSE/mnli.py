import os
# from util.data_loader import RParsedTextLField
# from util.data_loader import ParsedTextLField

from torchtext import data, vocab
from torchtext import datasets

import config
import torch


class MNLI(data.ZipDataset, data.TabularDataset):
    # url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    filename = 'multinli_0.9.zip'
    dirname = 'multinli_0.9'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, genre_field=None, root='.',
               train=None, validation=None, test=None):
        """Create dataset objects for splits of the SNLI dataset.
        This is the most flexible way to use the dataset.
        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose snli_1.0
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """
        path = cls.download_or_unzip(root)
        if parse_field is None:
            return super(MNLI, cls).splits(
                os.path.join(path, 'multinli_0.9_'), train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(MNLI, cls).splits(
            os.path.join(path, 'multinli_0.9_'), train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'genre': ('genre', genre_field)},
            filter_pred=lambda ex: ex.label != '-')

if __name__ == "__main__":
    pass