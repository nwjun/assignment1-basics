from collections import defaultdict
from typing import List, Tuple, Union

import regex as re


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, special_tokens: List[str]):
        self.special_tokens = special_tokens
        self.merges: List[Tuple[bytes, bytes]] = []  # index1, index2 => merged index
        self.word_dict = defaultdict(int)

        self.vocab = {x: bytes([x]) for x in range(256)}  # index -> bytes
        for i, tok in enumerate(special_tokens):
            self.vocab[256 + i] = tok.encode("utf-8")

    @staticmethod
    def read_from_file(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            all_text = f.read()
        return all_text

    def _compute_idx_freq(self, word_list: List[Union[List[int], int]]):
        idx_freq = defaultdict(int)
        for indices, occ in word_list:
            for idx_pair in zip(indices, indices[1:]):
                idx_freq[idx_pair] += occ
        return idx_freq

    def pretokenize(self, chunk: str):
        matches = re.finditer(self.PAT, chunk)
        for match in matches:
            indices = map(int, match.group().encode("utf-8"))
            self.word_dict[tuple(indices)] += 1
        return self.word_dict

    def merge(self, indices: List[int], pair: Tuple[int, int], new_index: int):
        i = 0
        new_indices = []

        while i < len(indices):
            if i + 1 < len(indices) and (indices[i], indices[i + 1]) == pair:
                new_indices.append(new_index)
                i += 2  # Skip over the two merged tokens
            else:
                new_indices.append(indices[i])
                i += 1

        return new_indices

    def train(self, vocab_size: int, corpus: str = "", input_path: str = ""):
        assert bool(corpus) ^ bool(
            input_path
        ), "Provide either corpus or input_path, not both"

        if input_path:
            corpus = self.read_from_file(input_path)

        if len(self.special_tokens) > 0:
            delimeter = "|".join(re.escape(tok) for tok in self.special_tokens)
            chunks = re.split(delimeter, corpus)
        else:
            # solve the case where no special_tokens and word will be splitted into single char
            chunks = [corpus]

        # there must be 2 loop each time, first to count, next to update

        for idx, chunk in enumerate(chunks):
            self.pretokenize(chunk)

        word_list = []
        for indices, occ in self.word_dict.items():
            word_list.append([list(indices), occ])

        new_index = len(self.vocab)
        while new_index < vocab_size:
            counts = self._compute_idx_freq(word_list)
            if not counts:
                break

            # choose lexicographically greater pair if tie
            # self.vocab[c] compare by bytes instead of index
            pair = max(
                counts.keys(),
                key=lambda p: (counts[p], [self.vocab[c] for c in p]),
            )
            index1, index2 = pair

            self.merges.append((self.vocab[index1], self.vocab[index2]))
            self.vocab[new_index] = self.vocab[index1] + self.vocab[index2]

            for idx, (indices, occ) in enumerate(word_list):
                new_indices = self.merge(indices, pair, new_index)
                word_list[idx] = [new_indices, occ]

            new_index += 1

        return self.vocab, self.merges


if __name__ == "__main__":
    corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    corpus = "This is the Hugging Face Course.<|endoftext|>This chapter is about tokenization.<|endoftext|>This section shows several tokenizer algorithms.<|endoftext|>Hopefully, you will be able to understand how they are trained and generate tokens."

    bpe_tokenizer = BPETokenizer(["<|endoftext|>"])
    vocab, merges = bpe_tokenizer.train(500, corpus=corpus)

    print(len(vocab))

    # print({token_id: repr(byte_val) for token_id, byte_val in vocab.items()})

    print("----------------")

    print(merges)
    for merge in merges:
        print(merge[0].decode("utf-8"), merge[1].decode("utf-8"))
