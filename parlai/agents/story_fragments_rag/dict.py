from transformers import RagTokenizer

from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent


class RAGDictionaryAgent(HuggingFaceDictionaryAgent):
    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        file_key = "facebook/rag-token-base"
        return RagTokenizer.from_pretrained(file_key)

    def _define_special_tokens(self, opt):
        self.start_token = '[CLS]'
        self.end_token = '[SEP]'
        self.null_token = '[PAD]'
        self.start_idx = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[
            0
        ]  # should be 101
        self.end_idx = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[
            0
        ]  # should be 102
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]  # should be 0

    def override_special_tokens(self, opt):
        self._define_special_tokens(opt)

        # set tok2ind for special tokens
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.null_token] = self.pad_idx
        # set ind2tok for special tokens
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.pad_idx] = self.null_token
