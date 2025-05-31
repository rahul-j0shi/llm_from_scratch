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
                lines = lines[1:]  # Skip the version line if present
            
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

