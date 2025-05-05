import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=768):
        super().__init__()
        self.model = base_model
        self.hidden_dim = hidden_dim
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward
