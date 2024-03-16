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

        # Adjusted input dimension: 768 (BERT's pooled output) + 2 (additional features) = 770
        self.evidence_classifier = nn.Linear(770, 4)  # Adjusted for 4 possible outputs for evidence
        self.suggestion_classifier = nn.Linear(770, 2)  # Adjusted for binary output for suggestion
        self.connection_classifier = nn.Linear(770, 2)  # Adjusted for binary output for connection
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sentiment=None, word_count=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output

        # Concatenate the sentiment feature to the pooled output
        if sentiment is not None and word_count is not None:
            pooled_output = torch.cat(
                (pooled_output, sentiment.unsqueeze(1), word_count.unsqueeze(1)), dim=1
            )

        # Apply each classifier to the pooled output
        evidence_logits = self.evidence_classifier(pooled_output)
        suggestion_logits = self.suggestion_classifier(pooled_output)
        connection_logits = self.connection_classifier(pooled_output)

        # Return a dictionary of logits
        return {
            'evidence': evidence_logits,
            'suggestion': suggestion_logits,
            'connection': connection_logits,  # Note: Renamed to 'connection' for consistency
        }
