import unittest
import torch
from multi_label_class import MultiTaskBertModel

class TestMultiTaskBertModel(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig()
        self.model = MultiTaskBertModel(self.config)

    def test_forward_with_labels(self):
        batch_size = 2
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, 128))
        attention_mask = torch.ones((batch_size, 128), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, 128), dtype=torch.long)
        evidence_labels = torch.randint(0, 4, (batch_size,))
        suggestion_labels = torch.randint(0, 2, (batch_size,))
        connection_labels = torch.randint(0, 2, (batch_size,))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            evidence_labels=evidence_labels,
            suggestion_labels=suggestion_labels,
            connection_labels=connection_labels,
        )

        self.assertIn("loss", outputs)
        self.assertTrue(outputs["loss"].requires_grad)

    def test_compute_metrics(self):
        batch_size = 2
        evidence_logits = torch.randn(batch_size, 4)
        suggestion_logits = torch.randn(batch_size, 2)
        connection_logits = torch.randn(batch_size, 2)
        evidence_labels = torch.randint(0, 4, (batch_size,))
        suggestion_labels = torch.randint(0, 2, (batch_size,))
        connection_labels = torch.randint(0, 2, (batch_size,))

        outputs = {
            "evidence": evidence_logits,
            "suggestion": suggestion_logits,
            "connection": connection_logits,
        }

        metrics = self.model.compute_metrics(
            outputs, evidence_labels, suggestion_labels, connection_labels
        )

        self.assertIn("evidence_acc", metrics)
        self.assertIn("evidence_f1", metrics)
        self.assertIn("suggestion_acc", metrics)
        self.assertIn("suggestion_f1", metrics)
        self.assertIn("connection_acc", metrics)
        self.assertIn("connection_f1", metrics)

if __name__ == "__main__":
    unittest.main()
