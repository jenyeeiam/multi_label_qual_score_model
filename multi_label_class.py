from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

from datasets import load_dataset, Dataset
import pandas as pd
import json


class MultiTaskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        # Define separate classifiers for each task
        self.evidence_classifier = nn.Linear(config.hidden_size, 4)  # For evidence score, 4 possible outputs
        self.suggestion_classifier = nn.Linear(config.hidden_size, 2)  # For suggestion, binary output
        self.connection_classifier = nn.Linear(config.hidden_size, 2)  # For linking, binary output

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled_output = outputs.pooler_output

        # Apply each classifier to the pooled output
        evidence_logits = self.evidence_classifier(pooled_output)
        suggestion_logits = self.suggestion_classifier(pooled_output)
        linking_logits = self.connection_classifier(pooled_output)

        return evidence_logits, suggestion_logits, linking_logits