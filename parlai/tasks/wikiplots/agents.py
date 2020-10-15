import more_itertools
from datasets import load_dataset

from parlai.core.teachers import DialogTeacher


class WikiPlotsDialogTeacher(DialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('WikiPlots Teacher Args')
        parser.add_argument(
            '--wikiplots-config-name',
            type=str,
            default="wikiplots_sentence",
            help="The Wikiplots huggingface configs name to load.",

        )
        parser.add_argument(
            '--wikiplots-train-split',
            type=int,
            default=80,
            help="The training split for Wikiplots.",
        )
        parser.add_argument(
            '--wikiplots-valid-split',
            type=int,
            default=10,
            help="The validation split for Wikiplots.",
        )
        parser.add_argument(
            '--wikiplots-test-split',
            type=int,
            default=10,
            help="The test split for Wikiplots.",
        )
        parser.add_argument(
            '--wikiplots-dataset-script-path',
            type=str,
            help="The Huggingface datasets path.",
        )

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']

        if opt['datatype'].startswith('train'):
            split = 'train'
        elif opt['datatype'].startswith('valid'):
            split = 'validation'
        else:
            split = 'test'

        self.id = 'wikiplots'
        self.split = split
        self.config_name = opt["wikiplots_config_name"]

        self.train_split = opt["wikiplots_train_split"]
        self.valid_split = opt["wikiplots_valid_split"]
        self.test_split = opt["wikiplots_test_split"]

        self.dataset_script = opt["wikiplots_dataset_script_path"]

        opt['datafile'] = ""

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        if self.split == "train":
            dataset = load_dataset(self.dataset_script, name=self.config_name, split=f'train[:{self.train_split}%]')
        elif self.split == "validation":
            dataset = load_dataset(self.dataset_script, name=self.config_name,
                                   split=f'train[{self.train_split}%:{self.train_split + self.valid_split}%]')
        else:
            dataset = load_dataset(self.dataset_script, name=self.config_name,
                                   split=f'train[-{self.test_split}%:]')

        for story in dataset:

            passage_pairs = more_itertools.chunked(story["passages"], n=2)
            passage_pairs = [p for p in passage_pairs if len(p) > 1 and p[1] is not None]
            passage_pairs_len = len(passage_pairs)
            for i, passage_pair in enumerate(passage_pairs):
                end_of_episode = i + 1 == passage_pairs_len
                yield {"text": passage_pair[0]["text"], "labels": passage_pair[1]["text"]}, end_of_episode


class DefaultTeacher(WikiPlotsDialogTeacher):
    pass
