from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
import spacy

class Multi30kDataset:
    def __init__(self, split='train',min_freq=2):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.min_freq = min_freq

        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split
        )

        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # ==========================================================
        # SPECIAL TOKENS
        # ==========================================================

        self.UNK_TOKEN = "<unk>"
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        self.special_tokens = [
            self.UNK_TOKEN,
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]

        # Build vocabulary only from train split
        if split == "train":
            self.build_vocab()

        # Process sentences
        self.process_data()
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
    
    def tokenize_de(self, text):

        return [
            token.text.lower()
            for token in self.spacy_de.tokenizer(text)
        ]
    def tokenize_en(self, text):

        return [
            token.text.lower()
            for token in self.spacy_en.tokenizer(text)
        ]
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        de_counter = Counter()
        en_counter = Counter()

        # Count words
        for sample in self.dataset:

            de_tokens = self.tokenize_de(sample["de"])
            en_tokens = self.tokenize_en(sample["en"])

            de_counter.update(de_tokens)
            en_counter.update(en_tokens)

        self.src_vocab = {}
        self.src_itos = {}

        for idx, token in enumerate(self.special_tokens):

            self.src_vocab[token] = idx
            self.src_itos[idx] = token

        idx = len(self.special_tokens)

        for word, freq in sorted(de_counter.items()):

            if freq >= self.min_freq:

                self.src_vocab[word] = idx
                self.src_itos[idx] = word

                idx += 1

        self.tgt_vocab = {}
        self.tgt_itos = {}

        for idx, token in enumerate(self.special_tokens):

            self.tgt_vocab[token] = idx
            self.tgt_itos[idx] = token

        idx = len(self.special_tokens)

        for word, freq in en_counter.items():

            if freq >= self.min_freq:

                self.tgt_vocab[word] = idx
                self.tgt_itos[idx] = word

                idx += 1

        # Vocabulary sizes
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab.stoi = self.src_vocab
        self.src_vocab.itos = self.src_itos

        self.tgt_vocab.stoi = self.tgt_vocab
        self.tgt_vocab.itos = self.tgt_itos
        return self.src_vocab, self.tgt_vocab
    def numericalize_de(self, tokens):

        return [
            self.src_vocab.get(
                token,
                self.src_vocab[self.UNK_TOKEN]
            )
            for token in tokens
        ]

    def numericalize_en(self, tokens):

        return [
            self.tgt_vocab.get(
                token,
                self.tgt_vocab[self.UNK_TOKEN]
            )
            for token in tokens
        ]

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        self.data = []

        for sample in self.dataset:

            
            de_tokens = self.tokenize_de(sample["de"])
            en_tokens = self.tokenize_en(sample["en"])

            de_tokens = (
                [self.SOS_TOKEN]
                + de_tokens
                + [self.EOS_TOKEN]
            )

            en_tokens = (
                [self.SOS_TOKEN]
                + en_tokens
                + [self.EOS_TOKEN]
            )

           
            src_ids = self.numericalize_de(de_tokens)
            tgt_ids = self.numericalize_en(en_tokens)

            self.data.append(
                (
                    torch.tensor(src_ids),
                    torch.tensor(tgt_ids)
                )
            )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]