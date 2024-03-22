from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from datasets import load_dataset, Dataset
import pandas as pd
import json


class MultiTaskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.evidence_classifier = nn.Linear(config.hidden_size, 4) # 4 possible outcomes
        self.suggestion_classifier = nn.Linear(config.hidden_size, 2) # binary outcome
        self.connection_classifier = nn.Linear(config.hidden_size, 2) # binary outcome
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, evidence_labels=None, suggestion_labels=None, connection_labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output

        evidence_logits = self.evidence_classifier(pooled_output)
        suggestion_logits = self.suggestion_classifier(pooled_output)
        connection_logits = self.connection_classifier(pooled_output)

        outputs = {
            "evidence": evidence_logits,
            "suggestion": suggestion_logits,
            "connection": connection_logits,
        }

        total_loss = 0.0
        if evidence_labels is not None:
            evidence_loss = F.cross_entropy(evidence_logits, evidence_labels)
            total_loss += evidence_loss
        if suggestion_labels is not None:
            suggestion_loss = F.binary_cross_entropy_with_logits(suggestion_logits, suggestion_labels.float())
            total_loss += suggestion_loss
        if connection_labels is not None:
            connection_loss = F.binary_cross_entropy_with_logits(connection_logits, connection_labels.float())
            total_loss += connection_loss

        outputs["loss"] = total_loss
        return outputs

    def compute_metrics(self, outputs, evidence_labels, suggestion_labels, connection_labels):
        evidence_logits = outputs["evidence"]
        suggestion_logits = outputs["suggestion"]
        connection_logits = outputs["connection"]

        evidence_preds = torch.argmax(evidence_logits, dim=1)
        suggestion_preds = (suggestion_logits > 0).long()
        connection_preds = (connection_logits > 0).long()

        evidence_acc = accuracy_score(evidence_labels.cpu(), evidence_preds.cpu())
        evidence_f1 = f1_score(evidence_labels.cpu(), evidence_preds.cpu(), average="macro")
        # Compute AUC for evidence task
        evidence_auc = roc_auc_score(
            evidence_labels.cpu(), evidence_logits.cpu(), multi_class="ovr", average="macro"
        )
        
        suggestion_acc = accuracy_score(suggestion_labels.cpu(), suggestion_preds.cpu())
        suggestion_f1 = f1_score(suggestion_labels.cpu(), suggestion_preds.cpu())
        suggestion_auc = roc_auc_score(
            suggestion_labels.cpu(), suggestion_logits.cpu(), average="binary"
        )
        
        connection_acc = accuracy_score(connection_labels.cpu(), connection_preds.cpu())
        connection_f1 = f1_score(connection_labels.cpu(), connection_preds.cpu())
        connection_auc = roc_auc_score(
            connection_labels.cpu(), connection_logits.cpu(), average="binary"
        )

        return {
            "evidence_acc": evidence_acc,
            "evidence_f1": evidence_f1,
            "evidence_auc": evidence_auc,
            "suggestion_acc": suggestion_acc,
            "suggestion_f1": suggestion_f1,
            "suggestion_auc": suggestion_auc,
            "connection_acc": connection_acc,
            "connection_f1": connection_f1,
            "connection_auc": connection_auc,
        }
