"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_2_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


class LLMDataset(Dataset):
    """
    A dataset for language modeling tasks.

    This class inherits from torch.utils.data.Dataset and implements a
    dataset that can load and preprocess data for language modeling. It
    takes a list of data dictionaries, a tokenizer, and optional prompt
    templates as input, and creates input ids, labels, and categories as
    output. The input ids and labels are padded and masked according to
    the tokenizer settings and the source and target lengths. The
    categories are encoded as integers using pandas.Categorical.

    Attributes:
        input_ids: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded input ids.
        labels: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded labels.
        categories: A list of integers representing the category codes.
        tokenizer: A transformers.PreTrainedTokenizer object that can
            encode and decode text.
    """
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"]):
        """
        Initializes the dataset with the given arguments.

        Args:
            list_data_dict: A list of dictionaries, each containing input,
                output, and optionally category keys and values as strings.
            tokenizer: A transformers.PreTrainedTokenizer object that can
                encode and decode text.
            prompt_input: An optional string template for creating the source
                text when the input key is present in the data dictionary.
                The template can use {input}, {output}, and {category} as
                placeholders for the corresponding values. The default value
                is PROMPT_DICT["prompt_input"].
            prompt_no_input: An optional string template for creating the
                source text when the input key is not present in the data
                dictionary. The template can use {output} and {category} as
                placeholders for the corresponding values. The default value is
                PROMPT_DICT["prompt_no_input"].
        """
        super(LLMDataset, self).__init__()

        #sources = [
        #    prompt_input.format_map(example) if example.get("input", "") != ""
        #    else prompt_no_input.format_map(example)
        #    for example in list_data_dict
        #]
        sources = [f"{example['input']}"
            for example in list_data_dict
                ]
        sources2 = [f"{example['category']}"
            for example in list_data_dict
                ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, sources2, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, strings, tokenizer):
        """
        Tokenizes a list of strings using the given tokenizer.

        Args:
            strings: A list of strings to be tokenized.
            tokenizer: A transformers.PreTrainedTokenizer object that can
                encode and decode text.

        Returns:
            A dictionary with the following keys and values:
                - input_ids: A list of torch.LongTensor objects of shape (
                    max_length,) containing the tokenized input ids.
                - labels: A list of torch.LongTensor objects of shape (
                    max_length,) containing the tokenized labels.
                - input_ids_lens: A list of integers representing the
                    lengths of the input ids before padding.
                - labels_lens: A list of integers representing the lengths of
                    the labels before padding.
        """
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, sources2, targets, tokenizer):
        """
        Preprocesses the sources and targets using the given tokenizer.

        Args:
            sources: A list of strings representing the source texts.
            targets: A list of strings representing the target texts.
            tokenizer: A transformers.PreTrainedTokenizer object that can
                encode and decode text.

        Returns:
            A dictionary with the following keys and values:
                - input_ids: A list of torch.LongTensor objects of shape (
                    max_length,) containing the padded input ids.
                - labels: A list of torch.LongTensor objects of shape (
                    max_length,) containing the padded labels.
        """
        examples = [s for s in sources]
        examples2 = [s for s in sources2]
        #examples = [s + t for s, t in zip(sources, targets)]
        #print(examples)
        #examples_tokenized, sources_tokenized = [
        #    self._tokenize_fn(example, tokenizer)
        #    for example in examples
        #]
        tokenized_list = [tokenizer(
                example,
                examples2[idx],
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation="only_first") for idx, example in enumerate(examples)]
        input_ids = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        #examples_tokenized, sources_tokenized = [
        #    self._tokenize_fn(strings, tokenizer)
        #    for strings in (examples, sources)
        #]
        #input_ids = examples_tokenized["input_ids"]
        labels = [0 if element == '0</s>' else 1 if element == '1</s>' else 2 for element in targets]#copy.deepcopy(input_ids)
        # labels = [0 if element == '0</s>' else 1 for element in targets]#copy.deepcopy(input_ids)
        # print("source_len")
        # 
        # 
        # source_len] = DefaultToken.IGNORE_INDEX.value
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])
