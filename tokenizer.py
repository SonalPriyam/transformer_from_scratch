import torch


class BPETokenizer:
    def __init__(self, vocab_size=300):
        assert vocab_size >= 260, "Vocab size must be at least 260 to fit special tokens and base bytes."
        self.vocab_size = vocab_size

        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
        }

        self.merges = {}
        self.vocab = {}
        self._init_base_vocab()

    def _init_base_vocab(self):
        self.vocab[self.PAD_IDX] = b"<PAD>"
        self.vocab[self.UNK_IDX] = b"<UNK>"
        self.vocab[self.SOS_IDX] = b"<SOS>"
        self.vocab[self.EOS_IDX] = b"<EOS>"

        for b in range(256):
            self.vocab[b + 4] = bytes([b])

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        raw_bytes = text.encode("utf-8")
        ids = [b + 4 for b in raw_bytes]

        num_merges = self.vocab_size - 260

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_token_id = 260 + i

            ids = self._merge(ids, best_pair, new_token_id)
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text, add_special_tokens=True):
        raw_bytes = text.encode("utf-8")
        ids = [b + 4 for b in raw_bytes]

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        if add_special_tokens:
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]

        return ids

    def decode(self, ids):
        byte_chunks = []
        for idx in ids:
            if idx in (self.PAD_IDX, self.SOS_IDX, self.EOS_IDX):
                continue
            byte_chunks.append(self.vocab.get(idx, b"<UNK>"))

        byte_sequence = b"".join(byte_chunks)
        return byte_sequence.decode("utf-8", errors="replace")

    def pad_sequence(self, ids, max_length):
        if len(ids) < max_length:
            return ids + [self.PAD_IDX] * (max_length - len(ids))
        return ids[:max_length]