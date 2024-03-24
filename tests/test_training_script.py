# test_training_script.py
import unittest
import torch
from multi_label_class import MultiTaskBertModel
from training_multi_label_script import initialize_model, train_model, evaluate_model


class TestTrainingScript(unittest.TestCase):
    def setUp(self):
        # Assuming CUDA is not available in the testing environment
        self.device = torch.device("cpu")
        self.model = initialize_model(self.device)
    
    def test_model_initialization(self):
        """Test that the model is initialized and moved to the correct device."""
        self.assertIsInstance(self.model, MultiTaskBertModel)
        # Check if model is on the correct device
        self.assertEqual(next(self.model.parameters()).device, self.device)
    
    def test_training_step_updates_weights(self):
        """Test that a single training step updates the model's weights."""
        # Mock dataloader with a single batch
        input_ids = torch.randint(0, 1000, (2, 128)).long()
        attention_mask = torch.ones(2, 128).long()
        token_type_ids = torch.zeros(2, 128).long()
        evidence_labels = torch.tensor([0, 1])
        suggestion_labels = torch.tensor([0, 1])
        connection_labels = torch.tensor([0, 1])
        
        train_dataloader = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                             'evidence': evidence_labels, 'suggestion': suggestion_labels, 'connection': connection_labels}]
        
        # Mock optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)  # Dummy scheduler
        
        initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        trained_model = train_model(
            self.model, self.device, train_dataloader, train_dataloader, 1, optimizer, scheduler
        )
        
        for name, param in trained_model.named_parameters():
            self.assertFalse(torch.equal(initial_state_dict[name], param), f"Parameter '{name}' did not update.")

# Add more tests here as needed

if __name__ == '__main__':
    unittest.main()
