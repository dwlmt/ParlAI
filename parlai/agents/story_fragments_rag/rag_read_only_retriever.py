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
        parser = parser.add_argument_group('RAG Lookup Retriever Arguments')

        parser.add_argument(
            '--rag-context-length',
            default=-1,
            type=int,
            help='Number of past utterances to remember when '
                 'building flattened batches of data in multi-'
                 'example episodes.',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'RagReadOnlyRetrieverAgent'

        self.use_cuda = torch.cuda.is_available()

        if shared is None:
            self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
            self.retriever = RagRetriever.from_pretrained("facebook/rag-token-base",
                                                          index_name="exact", use_dummy_dataset=True)

            self.decoder = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.retriever)

            if self.use_cuda:
                self.decoder = self.decoder.cuda()
        else:
            self.tokenizer = shared['tokenizer']
            self.retriever = shared['retriever']
            self.decoder = shared['decoder']

        self.context_length = opt["rag_context_length"]

        self.reset()

    def share(self):
        shared = super().share()
        shared['tokenizer'] = self.tokenizer
        shared['decoder'] = self.decoder
        shared['retriever'] = self.retriever
        return shared

    def reset(self):
        super().reset()
        self.episode_done = False
        self.current = []
        if self.context_length > 0:
            self.context = deque(maxlen=self.context_length)

    def train_act(self):

        obs = self.observation
        self.current.append(obs)
        self.episode_done = obs.get('episode_done', False)

        if self.episode_done:
            self.episode_done = False
            self.current.clear()
            self.context.clear()

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
                retrieved_outputs = self.decoder(input_ids, labels=None, output_retrieved=True)

                dl_scores = retrieved_outputs.doc_scores[0].tolist()
                dp_scores = retrieved_outputs.doc_scores.softmax(dim=-1)[0].tolist()
                doc_ids = retrieved_outputs.retrieved_doc_ids
                doc_dicts = self.retriever.index.get_doc_dicts(retrieved_outputs.retrieved_doc_ids)[0]

                if len(doc_dicts) > 0:
                    doc_texts = [f"{ti} - {te}" for ti, te in zip(doc_dicts["title"], doc_dicts["text"])]

                    reply['candidate_scores'] = dp_scores.tolist()
                    reply['candidate_ids'] = doc_ids.tolist()

                    reply['text_candidates'] = doc_texts
                    reply['text'] = doc_texts[0]

        return reply
