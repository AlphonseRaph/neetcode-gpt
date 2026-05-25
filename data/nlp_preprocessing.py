import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined_sentence = positive + negative

        all_words = []
        for sentence in combined_sentence:
            all_words.extend(sentence.split())

        id_word_dict = {word:i for i, word in enumerate(sorted(set(all_words)), start=1)}

        encoded_tensors = []
        for sentence in combined_sentence:
            encoded_sentence = [id_word_dict[w] for w in sentence.split()]

        #Convert to a tensor (specifying float to match return type)
            sentence_tensor = torch.tensor(encoded_sentence, dtype=torch.float32)
            encoded_tensors.append(sentence_tensor)
        
        padded_dataset = nn.utils.rnn.pad_sequence(
            encoded_tensors,
            batch_first=True, 
            padding_value=0, 
        )

        return padded_dataset
 