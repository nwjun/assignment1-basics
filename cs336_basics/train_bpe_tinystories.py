import os

from tokenization.bpe import BPETokenizer

if __name__ == "__main__":
    #  uv run scalene --html cs336_basics/train_bpe_tinystories.py
    TEST_RUN = False
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    dataset_path = "data/TinyStoriesV2-GPT4-train.txt"
    if TEST_RUN:
        dataset_path = "data/test_run.txt"
    save_vocab_path = f"artifacts/{os.path.basename(dataset_path)[:-3]}json"

    bpe_tokenizer = BPETokenizer(special_tokens)
    print("=== Start training bpe tokenizer ===")
    vocab, merges = bpe_tokenizer.train(vocab_size=vocab_size, input_path=dataset_path)
    print("=== Done training bpe tokenizer ===")

    # compare by len(vocab.values())
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token in vocab: ({len(longest_token)}) {longest_token}")
