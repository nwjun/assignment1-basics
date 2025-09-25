import ast
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import regex as re

TokDict = Dict[int, bytes]
MergeList = List[Tuple[bytes, bytes]]


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, special_tokens: Optional[Iterable[str]] = None):
        self.merges: MergeList = []  # index1, index2 => merged index
        self.vocab: TokDict = {x: bytes([x]) for x in range(256)}  # index -> bytes
        self._special_tokens: set[str] = set()
        self.special_tokens = special_tokens

    @property
    def special_tokens(self) -> set[str]:
        return self._special_tokens

    @special_tokens.setter
    def special_tokens(self, new_tokens: Iterable[str]):
        if not new_tokens:
            return

        new_tokens = set(new_tokens)
        self._special_tokens = self.special_tokens.union(set(new_tokens))

        reverse_vocab = {v: k for k, v in self.vocab.items()}
        for tok in new_tokens:
            encoded_tok = tok.encode("utf-8")
            if encoded_tok not in reverse_vocab:
                self.vocab[len(self.vocab)] = encoded_tok

    @staticmethod
    def read_txt_from_file(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            all_text = f.read()
        return all_text

    @staticmethod
    def save_vocab_merges(vocab: TokDict, merges: MergeList, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vocab_path = f"{path}_vocab.json"
        merge_path = f"{path}_merges.txt"

        # how vocab is built: self.vocab = {x: bytes([x]) for x in range(256)}
        # these are raw bytes, not guaranteed to be valid UTF-8 characters
        # latin-1 maps each byte directly to a Unicode character, wont throw errors
        vocab_serializable = {k: v.decode("latin1") for k, v in vocab.items()}
        with open(vocab_path, "w", encoding="utf-8") as vf:
            json.dump(vocab_serializable, vf, ensure_ascii=False, indent=2)

        with open(merge_path, "w", encoding="utf-8") as mf:
            for pair in merges:
                mf.write(f"{pair[0]} {pair[1]}\n")

    @staticmethod
    def load_vocab_merges_from_file(
        vocab_filepath: str, merges_filepath: str
    ) -> Tuple[TokDict, MergeList]:
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_serializable = json.load(vf)
            vocab = {int(k): v.encode("latin1") for k, v in vocab_serializable.items()}

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = []
            for line in mf:
                if line.startswith("#"):
                    continue

                # The line looks like: b' ' b't'
                # Find the space that separates the two byte representations
                # This is more robust than a simple split
                separator_index = line.find(" b'")
                if separator_index == -1:
                    continue

                part1_str = line[:separator_index].strip()
                part2_str = line[separator_index:].strip()

                try:
                    # ast.literal_eval will safely evaluate the string "b'...'"
                    # into a true bytes object
                    byte1 = ast.literal_eval(part1_str)
                    byte2 = ast.literal_eval(part2_str)
                    merges.append((byte1, byte2))
                except (ValueError, SyntaxError) as e:
                    print(
                        f"Warning: Could not parse merge line: {line.strip()}. Error: {e}"
                    )

        return vocab, merges

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output) and (optionally)
        a list of special tokens.
        """
        tokenizer = cls()
        tokenizer.vocab, tokenizer.merges = tokenizer.load_vocab_merges_from_file(
            vocab_filepath, merges_filepath
        )

        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

        return tokenizer

    def _compute_idx_freq(self, word_list: List[Union[List[int], int]]):
        idx_freq = defaultdict(int)
        for indices, occ in word_list:
            for idx_pair in zip(indices, indices[1:]):
                idx_freq[idx_pair] += occ
        return idx_freq

    def _preprocessing(self, corpus: str):
        """
        Preprocesses the input corpus by splitting it into chunks based on special tokens.

        Args:
            corpus (str): The input text to be preprocessed.

        Returns:
            list: A list of text chunks split by the special tokens. If no special tokens
                  are defined, the entire corpus is returned as a single chunk.
        """
        if len(self.special_tokens) > 0:
            delimeter = "|".join(re.escape(tok) for tok in self.special_tokens)
            chunks = re.split(delimeter, corpus)
        else:
            # solve the case where no special_tokens and word will be splitted into single char
            chunks = [corpus]
        return chunks

    def pretokenize(self, chunk: str, word_dict=None) -> Dict[Tuple[int, ...], int]:
        """
        Pre-tokenizes a given text chunk by finding matches based on a pattern,
        encoding the matches into UTF-8, and updating the word dictionary with
        the encoded indices.

        'hello' -> (104, 101, 108, 108, 111)
        (104, 101, 108, 108, 111): [occurence]

        Args:
            chunk (str): The input string to be pre-tokenized.

        Returns:
            dict: The word dictionary where keys are tuples of UTF-8
                  encoded indices and values are their corresponding counts.

        Note:
            - `self.PAT` is expected to be a regular expression pattern used
              for finding matches in the input chunk.
        """
        if not word_dict:
            word_dict = defaultdict(int)

        matches = re.finditer(self.PAT, chunk)
        for match in matches:
            indices = map(int, match.group().encode("utf-8"))
            word_dict[tuple(indices)] += 1
        return word_dict

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
            corpus = self.read_txt_from_file(input_path)

        chunks = self._preprocessing(corpus)

        word_dict = None
        for chunk in chunks:
            word_dict = self.pretokenize(chunk, word_dict)

        word_list = []
        for indices, occ in word_dict.items():
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

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        def encode_span(text: str) -> List[int]:
            """Encode a span of text that doesn't contain special tokens."""
            # Convert text to UTF-8 bytes
            utf8_bytes = text.encode("utf-8")

            # Check if entire span is already a token
            if utf8_bytes in byte2idx:
                return [byte2idx[utf8_bytes]]

            # Start with individual byte token IDs (idx)
            # Merges is using token ID, not byte val!
            # Looping utf8_bytes gives int, and we need byte to map to token ID
            ids = [byte2idx[bytes([byte_val])] for byte_val in utf8_bytes]

            # Apply merges greedily
            while True:
                # Find the best merge to apply
                best_pair = None
                best_rank = None

                for pair in zip(ids, ids[1:]):
                    if pair in rank:
                        pair_rank = rank[pair]
                        if best_rank is None or pair_rank < best_rank:
                            best_pair = pair
                            best_rank = pair_rank

                if best_pair is None:
                    break

                # Apply the merge
                merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                new_token_id = byte2idx[merged_bytes]
                ids = self.merge(ids, best_pair, new_token_id)

            return ids

        # Create proper byte-to-token-id mapping
        byte2idx: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Create merge ranking - maps (token_id1, token_id2) -> merge_order
        rank = {}
        for i, (bytes1, bytes2) in enumerate(self.merges):
            idx1 = byte2idx[bytes1]
            idx2 = byte2idx[bytes2]
            rank[(idx1, idx2)] = i

        # Handle special tokens by splitting text first
        if self.special_tokens:
            delimiter = "|".join(re.escape(tok) for tok in self.special_tokens)
            splitter = re.compile(f"({delimiter})")
            chunks = splitter.split(text)
        else:
            chunks = [text]

        encoded_list: List[int] = []

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                # Handle special tokens
                special_bytes = chunk.encode("utf-8")
                encoded_list.append(byte2idx[special_bytes])
            elif chunk:  # Only process non-empty chunks
                # Apply regex pattern to find tokenizable spans
                for match in re.finditer(self.PAT, chunk):
                    span = match.group()
                    encoded_list.extend(encode_span(span))

        return encoded_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files
        that cannot be directly loaded into memory.
        """
        for text_chunk in iterable:
            for token_id in self.encode(text_chunk):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        decoded_bytes = bytearray()

        for token_id in ids:
            if token_id in self.vocab:
                decoded_bytes.extend(self.vocab[token_id])
            else:
                # Use replacement character for unknown token IDs
                decoded_bytes.extend(b"\xef\xbf\xbd")  # UTF-8 replacement character

        return decoded_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    save_path = "artifacts/test_run"
    bpe_tokenizer = BPETokenizer(["<|endoftext|>"])
    vocab, merges = bpe_tokenizer.train(500, corpus=corpus)

    bpe_tokenizer.save_vocab_merges(vocab, merges, save_path)
    loaded_vocab, loaded_merges = bpe_tokenizer.load_vocab_merges_from_file(
        save_path + "_vocab.json", save_path + "_merges.txt"
    )

    # Test Unicode
    print("Testing Unicode...")
    test_string = "🙃"
    encode_output = bpe_tokenizer.encode(test_string)
    decoded = bpe_tokenizer.decode(encode_output)
    print(f"Original: {test_string}")
    print(f"Encoded: {encode_output}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_string == decoded}")

    # Test mixed Unicode and ASCII
    test_string2 = "Hello 🙃 World"
    encode_output2 = bpe_tokenizer.encode(test_string2)
    decoded2 = bpe_tokenizer.decode(encode_output2)
    print(f"\nOriginal: {test_string2}")
    print(f"Encoded: {encode_output2}")
    print(f"Decoded: {decoded2}")
    print(f"Match: {test_string2 == decoded2}")
