import json
import os
import re

from tqdm.auto import tqdm
from multiprocessing import cpu_count, Pool

# === ГЛОБАЛЬНАЯ ПРОВЕРКА C++ ===
CPP_AVAILABLE = False
try:
    from fast_bpe_cpp import BPETokenizer
    CPP_AVAILABLE = True
    # print('asdas')
except Exception as e:
    # print('1454564')
    pass  # Молча отключаем C++

# === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ PYTHON-МУЛЬТИПРОЦЕССИНГА ===
def _tokenize_block_mp(args):
    text_block, vocab_dict = args
    from Vocab import Vocab

    class BPELocal:
        def __init__(self, vocab_dict):
            self.vocab = Vocab(vocab_dict)
            self.unk_token_id = 1
            self._build_trie()

        def _build_trie(self):
            self.trie = {}
            for token, idx in self.vocab.xtoi.items():
                if not isinstance(token, str):
                    continue
                node = self.trie
                for ch in token:
                    if ch not in node:
                        node[ch] = {}
                    node = node[ch]
                node["__id__"] = idx

        def encode(self, text):
            ids = []
            i = 0
            n = len(text)
            unk_id = self.unk_token_id
            while i < n:
                node = self.trie
                best_match_len = 0
                best_match_id = unk_id
                j = i
                while j < n and text[j] in node:
                    node = node[text[j]]
                    if "__id__" in node:
                        best_match_len = j - i + 1
                        best_match_id = node["__id__"]
                    j += 1
                if best_match_len > 0:
                    ids.append(best_match_id)
                    i += best_match_len
                else:
                    ids.append(unk_id)
                    i += 1
            return ids

    tokenizer = BPELocal(vocab_dict)
    return tokenizer.encode(text_block)


class BPE:
    def __init__(self, vocab=None):
        self.show_tqdm = False
        self.show_tqdm_on_encode = False
        self.mean_length = 3

        self.parallel_threshold = 1_000_000
        self.parallel_block_size = 1_000_000
        self.max_workers = min(23, cpu_count() - 1)
        self.cpp_threshold = 50_000_000

        base_vocab = {0: '<[PAD]>', 1: '<[UNK]>', 2: '<[INS]>'}

        if vocab is None:
            from Vocab import Vocab as VocabClass
            self.vocab = VocabClass(base_vocab)
        else:
            self.vocab = vocab
            for k, v in base_vocab.items():
                if k not in self.vocab.itox:
                    self.vocab.itox[k] = v
            self.vocab.xtoi = {token: idx for idx, token in self.vocab.itox.items()}

        self.mask_token_id = -100
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.ins_token_id = 2

        # === Инициализация C++ (если доступен) ===
        self._cpp_tokenizer = None
        self._vocab_dict = None
        global CPP_AVAILABLE
        if CPP_AVAILABLE:
            try:
                self._vocab_dict = {token: idx for token, idx in self.vocab.xtoi.items()}
                self._cpp_tokenizer = BPETokenizer(self._vocab_dict, self.unk_token_id)
                print("✅ Используется C++ токенизатор")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации C++: {e}")
                CPP_AVAILABLE = False

        if vocab is None and os.path.exists('bpe_vocab.json'):
            self.load()
        elif not CPP_AVAILABLE:
            self._build_trie()

    def __call__(self, *args, **kwargs):
        ret = []
        for arg in args:
            ret.extend(self.encode(arg, **kwargs))
        return ret

    def __len__(self):
        return len(self.vocab)

    @property
    def len(self):
        return len(self.vocab)

    @property
    def ids(self):
        return self.vocab.xtoi

    def encode(self, text, do_tqdm=True):
        if not isinstance(text, str):
            return self.vocab[text]

        if CPP_AVAILABLE and self._cpp_tokenizer is not None:
            if len(text) >= self.cpp_threshold:
                return self._encode_cpp_parallel(text)
            else:
                return self._cpp_tokenizer.encode(text)

        if len(text) >= self.parallel_threshold:
            n_blocks = max(1, len(text) // self.parallel_block_size)
            n_workers = min(self.max_workers, n_blocks)
            return self._encode_parallel(
                text,
                max_workers=n_workers,
                min_block_size=self.parallel_block_size
            )
        else:
            return self._encode_single(text, do_tqdm=do_tqdm)

    def _encode_cpp_parallel(self, text):
        from concurrent.futures import ProcessPoolExecutor
        n_workers = min(self.max_workers, (len(text) + self.cpp_threshold - 1) // self.cpp_threshold)
        block_size = len(text) // n_workers
        blocks = []
        for i in range(n_workers):
            start = i * block_size
            end = start + block_size if i < n_workers - 1 else len(text)
            blocks.append(text[start:end])

        tasks = [(blk, self._vocab_dict, self.unk_token_id) for blk in blocks]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_cpp_tokenize_block, tasks))

        final_ids = []
        for ids in results:
            final_ids.extend(ids)
        return final_ids

    def _build_trie(self):
        self.trie = {}
        for token, idx in self.vocab.xtoi.items():
            if not isinstance(token, str):
                continue
            node = self.trie
            for ch in token:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node["__id__"] = idx

    def _encode_single(self, text, do_tqdm=True):
        ids = []
        i = 0
        n = len(text)
        unk_id = self.unk_token_id

        if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
            pbar = tqdm(total=n, desc="Encoding")
            last_i = 0

        while i < n:
            node = self.trie
            best_match_len = 0
            best_match_id = unk_id
            j = i
            while j < n and text[j] in node:
                node = node[text[j]]
                if "__id__" in node:
                    best_match_len = j - i + 1
                    best_match_id = node["__id__"]
                j += 1

            if best_match_len > 0:
                ids.append(best_match_id)
                i += best_match_len
            else:
                ids.append(unk_id)
                i += 1

            if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
                pbar.update(i - last_i)
                last_i = i

        if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
            pbar.close()

        return ids

    def _encode_parallel(self, text: str, max_workers: int = 23, min_block_size: int = 1_000_000):
        n_workers = max(1, min(max_workers, (len(text) + min_block_size - 1) // min_block_size))
        block_size = len(text) // n_workers
        blocks = []
        for i in range(n_workers):
            start = i * block_size
            end = start + block_size if i < n_workers - 1 else len(text)
            blocks.append(text[start:end])

        vocab_dict = self.vocab.itox
        tasks = [(block, vocab_dict) for block in blocks]

        with Pool(processes=n_workers) as pool:
            results = pool.map(_tokenize_block_mp, tasks)

        final_ids = []
        for ids in results:
            final_ids.extend(ids)
        return final_ids

    def split(self, text, do_tqdm=False):
        if not isinstance(text, str):
            raise ValueError("split expects a string")

        tokens = []
        i = 0
        n = len(text)

        if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
            pbar = tqdm(total=n, desc="Splitting")
            last_i = 0

        while i < n:
            node = self.trie
            best_match_len = 0
            best_match_token = '<[UNK]>'
            j = i
            while j < n and text[j] in node:
                node = node[text[j]]
                if "__id__" in node:
                    best_match_len = j - i + 1
                    token_id = node["__id__"]
                    best_match_token = self.vocab.itox[token_id]
                j += 1

            if best_match_len > 0:
                tokens.append(best_match_token)
                i += best_match_len
            else:
                tokens.append('<[UNK]>')
                i += 1

            if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
                pbar.update(i - last_i)
                last_i = i

        if (self.show_tqdm or self.show_tqdm_on_encode) and do_tqdm:
            pbar.close()

        return tokens

    def decode(self, ids):
        try:
            text = "".join(self.vocab.itox.get(i, '<[UNK]>') for i in ids)
            # text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text
        except Exception as e:
            print('Выхлоп:', ids)
            raise e

    def load(self, path='bpe_vocab.json'):
        from Vocab import Vocab as VocabClass
        self.vocab = VocabClass(path)
        base_ids = {0, 1, 2}
        for bid in base_ids:
            if bid not in self.vocab.itox:
                default_tokens = {0: '<[PAD]>', 1: '<[UNK]>', 2: '<[INS]>'}
                self.vocab.itox[bid] = default_tokens[bid]
        self.vocab.xtoi = {token: idx for idx, token in self.vocab.itox.items()}
        if CPP_AVAILABLE:
            self._vocab_dict = {token: idx for token, idx in self.vocab.xtoi.items()}
            self._cpp_tokenizer = BPETokenizer(self._vocab_dict, self.unk_token_id)
        else:
            self._build_trie()
        if self.vocab.itox:
            self.mean_length = sum(len(token) for token in self.vocab.itox.values()) / len(self.vocab.itox)
        else:
            self.mean_length = 3

    def train(self, text=''):
        raise NotImplementedError("Training logic is not included.")


# === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ C++ МУЛЬТИПРОЦЕССИНГА ===
def _cpp_tokenize_block(args):
    text_block, vocab_dict, unk_id = args
    tokenizer = BPETokenizer(vocab_dict, unk_id)
    return tokenizer.encode(text_block)