#!/usr/bin/env python3
from collections import deque

import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

from parlai.core.agents import Agent


class RagReadOnlyRetrieverAgent(Agent):
    """ RagReadOnlyRetrieverAgent

    A simple test agent that just returns the closest match from a RAG index. Mainly for just testing how
    well lookup with work using the RAG/DPR lookup for narratives.

    """

    @staticmethod
    def add_cmdline_args(parser):

        def add_bool_arg(parser, name, default=False):
            name_underscore = name.replace("-", "_")
            parser.add_argument(f'--{name}', dest=name_underscore, action='store_true')
            parser.add_argument(f'--no-{name}', dest=name_underscore, action='store_false')
            parser.set_defaults(**{name_underscore: default})

        parser = parser.add_argument_group('RAG Arguments')

        parser.add_argument(
            '--rag-pretrained-model',
            default="facebook/rag-token-nq",
            type=str,
            help='The pretrained model to use.',
        )

        add_bool_arg(parser, 'rag-use-dummy-dataset', default=True)
        add_bool_arg(parser, 'rag-output-retrieved', default=True)

        parser.add_argument(
            '--rag-n-docs',
            default=5,
            type=int,
            help='Number of documents to retrieve.',
        )
        parser.add_argument(
            '--rag-max-combined-length',
            default=300,
            type=int,
            help='Max length of contextualized input returned by the retriever.',
        )
        parser.add_argument(
            '--rag-retrieval-vector-size',
            default=768,
            type=int,
            help='Size of the encoded knowledgebase vectors. ',
        )
        parser.add_argument(
            '--rag-dataset',
            default="wiki_dpr",
            type=str,
            help='A dataset identifier for Huggingface datasets to use as the knowledgebase, default "wiki_dpr"',
        )
        parser.add_argument(
            '--rag-index-name',
            default="exact",
            type=str,
            help='The index name for Faiss, "compressed" is a bucketed index, "exact" is an exact match index.',
        )
        parser.add_argument(
            '--rag-index-path',
            required=False,
            type=str,
            help='Optional path to the Faiss index to load from disk.',
        )
        parser.add_argument(
            '--history-size',
            default=1,
            type=int,
            help='Number of context history states to retain.',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'RagReadOnlyRetrieverAgent'

        self.use_cuda = torch.cuda.is_available()

        if shared is None:

            pretrained_model_name = opt['rag_pretrained_model']

            self.config_override = {}
            self.config_override['use_dummy_dataset'] = opt['rag_use_dummy_dataset']
            self.config_override['n_docs'] = opt['rag_n_docs']
            self.config_override['max_combined_length'] = opt['rag_max_combined_length']
            self.config_override['dataset'] = opt['rag_dataset']
            self.config_override['index_name'] = opt['rag_index_name']
            self.config_override['retrieval_vector_size'] = opt['rag_retrieval_vector_size']
            self.config_override['output_retrieved'] = opt['rag_output_retrieved']

            if 'rag_index_path' in opt and opt['rag_index_path'] is not None:
                self.config_override['index_path'] = opt['rag_index_path']

            self.tokenizer = RagTokenizer.from_pretrained(pretrained_model_name)
            self.retriever = RagRetriever.from_pretrained(pretrained_model_name,
                                                          **self.config_override)

            self.model = RagSequenceForGeneration.from_pretrained(pretrained_model_name, retriever=self.retriever,
                                                                  **self.config_override)

            if self.use_cuda:
                self.model = self.model.cuda()
        else:
            self.tokenizer = shared['tokenizer']
            self.retriever = shared['retriever']
            self.model = shared['model']
            self.config_override = shared['config_override']

        self.history_size = opt['history_size']

        self.reset()

    def share(self):
        shared = super().share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model
        shared['retriever'] = self.retriever
        shared['config_override'] = self.config_override
        return shared

    def reset(self):
        super().reset()
        self.episode_done = False
        self.current = []
        self.history = deque(maxlen=self.history_size)

    def train_act(self):

        obs = self.observation
        self.history.append(obs)
        self.episode_done = obs.get('episode_done', False)

        if self.episode_done:
            self.episode_done = False
            self.current.clear()
            self.history.clear()

        return {'id': self.getID(), 'text': obs.get('labels', ['I don\'t know'])[0]}

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        if 'labels' in obs:
            return self.train_act()
        if 'text' in obs:

            input_dict = self.tokenizer.prepare_seq2seq_batch(obs["text"],
                                                              return_tensors="pt")
            input_ids = input_dict["input_ids"]

            with torch.no_grad():

                if self.use_cuda:
                    input_ids = input_ids.cuda()

                # retrieve support docs
                retrieved_outputs = self.model(input_ids, labels=None, output_retrieved=True)

                # Get the document output.
                dl_scores = retrieved_outputs.doc_scores[0].tolist()
                dp_scores = retrieved_outputs.doc_scores.softmax(dim=-1)[0]
                doc_ids = retrieved_outputs.retrieved_doc_ids
                doc_dicts = self.retriever.index.get_doc_dicts(retrieved_outputs.retrieved_doc_ids)[0]

                # Use as a generator to produce the base text.
                generated = self.model.generate(input_ids=input_ids)
                generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

                if len(doc_dicts) > 0:
                    doc_texts = [f"{ti} - {te}" for ti, te in zip(doc_dicts["title"], doc_dicts["text"])]

                    reply['candidate_scores'] = dp_scores.tolist()
                    reply['candidate_ids'] = doc_ids.tolist()

                    reply['text_candidates'] = doc_texts
                    reply['text'] = " \n ".join([generated_text] + doc_texts)

        return reply
