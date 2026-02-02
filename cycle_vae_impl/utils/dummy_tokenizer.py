class DummyTokenizer:
    """Minimal tokenizer shim for offline/synthetic runs.

    Only provides the fields used by this implementation.
    """

    def __init__(self, vocab_size: int, mask_token_id: int = 103, pad_token_id: int = 0):
        self._vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)

    def __len__(self):
        return self._vocab_size
