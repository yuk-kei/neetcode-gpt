import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        words = set()
        combined = positive + negative
        
        for sentence in combined:
            for word in sentence.split():
                words.add(word)
        
        sorted_list = sorted(list(words))
        
        dictionary = {word: i + 1 for i, word in enumerate(sorted_list)}
        
        def encode(sentence: str) -> List[int]:
            int_values = []
            for word in sentence.split():
                int_values.append(dictionary[word])
            return int_values
        
        tensor_list = []
        
        for sentence in combined:
            tensor_list.append(torch.tensor(encode(sentence)))
            
        return nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)        
