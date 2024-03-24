import unittest
import torch
from multi_label_class import MultiTaskBertModel
from transformers import BertConfig

class TestMultiTaskBertModel(unittest.TestCase):
    def setUp(self):
        self.model = MultiTaskBertModel(BertConfig())
    
    def test_forward_output_shapes(self):
        # Mock inputs
        input_ids = torch.randint(0, 1000, (50, 128)).long()  # Batch size of 2 and sequence length of 128
        attention_mask = torch.ones(50, 128).long()
        token_type_ids = torch.zeros(50, 128).long()


        # Forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Test shapes
        self.assertEqual(outputs['evidence'].shape, (50, 4)) # Assuming 4 possible outcomes for evidence
        self.assertEqual(outputs['suggestion'].shape, (50, 1)) # Assuming binary outcome, output shape adjusted for your model specifics
        self.assertEqual(outputs['connection'].shape, (50, 1)) # Assuming binary outcome, output shape adjusted for your model specifics

    def test_compute_metrics(self):
        # Adjusted mock outputs and labels for testing compute_metrics with a more significant test set
        mock_outputs = {
            "evidence": torch.randn(100, 4),  # Random scores for 100 samples across 4 classes
            "suggestion": torch.randn(100, 1),  # Random scores for binary outcome, 100 samples
            "connection": torch.randn(100, 1)  # Random scores for binary outcome, 100 samples
        }
        # Adjusting the labels to match the batch size of 100
        evidence_labels = torch.randint(0, 4, (100,))  # Random class labels for 100 samples
        suggestion_labels = torch.randint(0, 2, (100,))  # Binary labels for 100 samples
        connection_labels = torch.randint(0, 2, (100,))  # Binary labels for 100 samples

        # Compute metrics
        metrics = self.model.compute_metrics(mock_outputs, evidence_labels, suggestion_labels, connection_labels)

        # Check if metrics are returned and have expected keys
        self.assertIsInstance(metrics, dict)
        self.assertIn("evidence_acc", metrics)
        self.assertIn("evidence_f1", metrics)
        self.assertIn("evidence_auc", metrics)
        self.assertIn("suggestion_acc", metrics)
        self.assertIn("suggestion_f1", metrics)
        self.assertIn("suggestion_auc", metrics)
        self.assertIn("connection_acc", metrics)
        self.assertIn("connection_f1", metrics)
        self.assertIn("connection_auc", metrics)

        # Further checks can be added based on the expected ranges or properties of these metrics

if __name__ == '__main__':
    unittest.main()
