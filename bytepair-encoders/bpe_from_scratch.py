from collections import Counter, deque
from functools import lru_cache
import json

class BPETokenizerSimple:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i!=0:
                processed_text.append("G")
            if char != " ":
                processed_text.append(char)
        
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
        if "G" not in unique_chars:
            unique_chars.append("G")
        
        self.vocab = {i:char for i,char in enumerate(unique_chars)}
        self.inverse_vocab = {char:i for i,char in self.vocab.items()}

        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    idx = len(self.vocab)
                    self.vocab[idx] = token
                    self.inverse_vocab[token] = idx
            
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode = "most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
        
        for (p0,p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
    
    def load_vocab_and_merges_openai(self, vocab_path, bpe_merges_path):
        with open(vocab_path, 'r', encoding='utf-8') as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}
        
        if "/n" not in self.inverse_vocab:
            fallback_token = next((token for token in ["<|endoftext|>", "Ġ", ""] if token in self.inverse_vocab), None)
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                raise KeyError("No suitable newline token found in the vocabulary.")
    
        self.bpe_ranks = {}
        with open(bpe_merges_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
            
            rank =0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair)==2:
                    token1,token2 = pair
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                    else:
                        print(f"Skipping pair {pair} as one or both tokens are not in the vocabulary.")
    
    def encode(self,text):
        tokens = []
        lines = text.split("/n")
        for i,line in enumerate(lines):
            if i>0:
                tokens.append("\n")
            words = line.split("\n")
            for j,word in enumerate(words):
                if j==0:
                    if i>0:
                        tokens.append("G"+word)
                    else:
                        tokens.append(word)
                else:
                    tokens.append("G"+word)
        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                sub_token_ids = self.tokenize_with_bpe(token)
                token
        return token_ids

    def tokenize_with_bpe(self, token):
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        i += 2
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        symbols = [self.vocab[id_num] for id_num in token_ids]

        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            if bigram is None or bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
