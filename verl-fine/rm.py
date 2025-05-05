import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, preferred, dispreferred = self.data[idx]

        preferred_tokens = self.tokenizer(prompt + preferred, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        dispreferred_tokens = self.tokenizer(prompt + dispreferred, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        return {
            "preferred_input_ids": preferred_tokens["input_ids"].squeeze(),
            "preferred_attention_mask": preferred_tokens["attention_mask"].squeeze(),
            "dispreferred_input_ids": dispreferred_tokens["input_ids"].squeeze(),
            "dispreferred_attention_mask": dispreferred_tokens["attention_mask"].squeeze()
        }

data = [
    ("Translate to French: Hello, how are you?", "Bonjour, comment allez-vous?", "Bonjour, comment vas-tu?"),  
    ("Write a short poem about the ocean.", "The ocean vast, a calming blue, Waves crashing softly, ever true.", "Ocean strong, waves crash hard."), 
    ("Summarize the plot of Hamlet.", "Hamlet is a tragedy about revenge, loss, and madness.", "Hamlet dies at the end."), 
]




class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.model.config
        self.transformer = self.model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        rewards = self.v_head(last_hidden_state[:, -1, :]).squeeze(-1)
        return rewards

def train_reward_model(reward_model, dataset, epochs=3, batch_size=4, learning_rate=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(reward_model.parameters(), lr=learning_rate)

    reward_model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            preferred_input_ids = batch["preferred_input_ids"].to(reward_model.device)
            preferred_attention_mask = batch["preferred_attention_mask"].to(reward_model.device)
            dispreferred_input_ids = batch["dispreferred_input_ids"].to(reward_model.device)
            dispreferred_attention_mask = batch["dispreferred_attention_mask"].to(reward_model.device)

            reward_preferred = reward_model(preferred_input_ids, attention_mask=preferred_attention_mask)
            reward_dispreferred = reward_model(dispreferred_input_ids, attention_mask=dispreferred_attention_mask)

            loss = -torch.log(torch.sigmoid(reward_preferred - reward_dispreferred)).mean()

            optimizer.zero_grad()