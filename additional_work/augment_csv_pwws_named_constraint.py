from textattack.transformations import WordSwap

# Import the model
import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper

# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification
# goal_function = UntargetedClassification(model_wrapper)

# Import the dataset
from textattack.datasets import HuggingFaceDataset
# dataset = HuggingFaceDataset("ag_news", None, "test")

from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack import Attack

from tqdm import tqdm # tqdm provides us a nice progress bar.
from textattack.loggers.csv_logger import CSVLogger # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs
from textattack.datasets import Dataset

import numpy as np
import pandas as pd
import tqdm


import torch
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
import os

model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="pytorch_model.bin", config="config_pyt.json")

num_labels = 3

model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)

tokenizer = BertTokenizer.from_pretrained(os.getcwd())

model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

goal_function = UntargetedClassification(model_wrapper)


# The below is a pre-transformation Constraint
from textattack.constraints.pre_transformation_constraint import PreTransformationConstraint
class NoTGTEntityWordSwap(PreTransformationConstraint):
    """ If the word is the special token TARGET__ENTITY, then do not replace that word.
    """
    
    # We don't need a constructor, since our class doesn't require any parameters.
    def __init__(self, target_entity="TARGET__ENTITY"):
        self.target_entity = target_entity
        
    def _get_modifiable_indices(self, current_text):
        """ Check if the word is target_entity, then return the word as it is
        """
        text_words = current_text.words
        target_entity_indices = []
        for idx in range(len(text_words)):
            if  self.target_entity in text_words[idx]:
                target_entity_indices.append(idx)
    
        modifiable_idx = set(range(len(current_text.words)))
        for remove_idx in target_entity_indices:
            modifiable_idx.remove(remove_idx)
        return modifiable_idx

from textattack.attack_recipes.pwws_ren_2019 import PWWSRen2019

attack = PWWSRen2019.build(model_wrapper)

from textattack.transformations import CompositeTransformation
from textattack.augmentation import Augmenter

transformation = CompositeTransformation([attack.transformation])
constraints = attack.constraints
constraints.append(NoTGTEntityWordSwap())


augmenter = Augmenter(transformation=transformation,
                      constraints=constraints,
                      pct_words_to_swap=0.2,
                      transformations_per_example=1)

data_path = "data/fixed_test.csv"

data = pd.read_csv(data_path)

class InputExample(object):
    """
        Training / test example for masked word prediction and author sentiment classification.
    """

    def __init__(self, masked_sentence, original_sentence, sentiment, target_entity):
        """
        Construct and InputExample

        Args:
            masked_sentence (str):
                A string containing the input article with target_entity masked.

            original_sentence (str):
                A string containing the input article with no masks.

            sentiment (str):
                Author's sentiment
        """
        self.masked_sentence = masked_sentence
        self.original_sentence = original_sentence
        self.sentiment = sentiment
        self.target_entity = target_entity

def convert_text_to_examples(masked_texts, original_texts, labels, target_entities):
    """
        Create InputExamples.
    """
    InputExamples = []

    for masked_text, original_text, label, target_entity in zip(masked_texts, original_texts, labels, target_entities):
        InputExamples.append(
            InputExample(masked_text, original_text, label, target_entity)
        )
    return InputExamples

def replace_named_entities(input_examples):
    for ex in input_examples:
        named_entity = ex.target_entity
#         print(f"Using Named Entity {named_entity}")
        tgt_parts = named_entity.split()
#         print(f"Tokenized Named Entity {tgt_parts}")
        count = 0
        for part in tgt_parts:
            ex.original_sentence = ex.original_sentence.replace(part, f"TARGET__ENTITY_{count}")
            count += 1
#         break
    return input_examples

def restore_named_entities(input_example, target_entity):
    tgt_parts = target_entity.split()
    for part_idx in range(len(tgt_parts)):
        input_example = input_example.replace(f"TARGET__ENTITY_{part_idx}", tgt_parts[part_idx])
    return input_example


# donald_trump_input_examples = convert_text_to_examples(data['MASKED_DOCUMENT'], data['DOCUMENT'], data['TRUE_SENTIMENT'], data['TARGET_ENTITY'], "Donald Trump")
# donald_trump_replaced_examples = replace_named_entities(donald_trump_input_examples)
# print(donald_trump_replaced_examples[0].original_sentence)

input_examples = convert_text_to_examples(data['MASKED_DOCUMENT'], data['DOCUMENT'], data['TRUE_SENTIMENT'], data['TARGET_ENTITY'])
replaced_input_examples = replace_named_entities(input_examples)

replaced_input_examples_list = []

for ex in replaced_input_examples:
    replaced_input_examples_list.append(ex.original_sentence)

def get_aug_doc_list(doc):
    augmented_doc = []
    para_list = []

    sent_lst = doc.split(".")
    augmented_doc = augmenter.augment_many(sent_lst, True)
    para_list = [""]*len(augmented_doc[0])

    for sent_list in augmented_doc:
        if len(para_list) != len(sent_list):
            continue
        for idx in range(len(sent_list)):
            if para_list[idx] == "":
                para_list[idx] = sent_list[idx]
            else:
                para_list[idx] = f"{para_list[idx]}. {sent_list[idx]}"
    return para_list

for doc_idx in range(len(replaced_input_examples_list)):
    print(f"=========== DOC IDX {doc_idx} =========")
    doc = replaced_input_examples_list[doc_idx]

    doc_aug_para_list = get_aug_doc_list(doc)

    for aug_doc in doc_aug_para_list:
        orig_doc_row = data.loc[doc_idx]
        aug_doc = restore_named_entities(aug_doc, orig_doc_row["TARGET_ENTITY"])
        orig_doc_row["DOCUMENT"] = aug_doc
#        print(orig_doc_row)
        data = data.append(orig_doc_row, ignore_index=True)
        data.to_csv('fixed_data_pwws_aug_named_constrained.csv')
