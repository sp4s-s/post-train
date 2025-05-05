import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torchvision


# HydraConfig = {}

sft_model = AutoModelForCausalLM.from_pretrained()  
sft_tokenizer = AutoTokenizer.from_pretrained()

class RewardModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(sft_tokenizer.vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8), num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        output = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        pooled_output = output.mean(dim=1)  # Simple average pooling for sequence representation
        reward = self.fc(pooled_output)
        return reward.squeeze(1)


class MultimodalRewardModel(nn.Module): #Template piece
    def __init__(self, text_reward_model, image_encoder, hidden_size=768):
        super().__init__()
        self.text_reward_model = text_reward_model
        self.image_encoder = image_encoder
        self.fusion = nn.Linear(hidden_size * 2, hidden_size) 
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, input_ids, attention_mask, image_features):
        text_reward = self.text_reward_model(input_ids, attention_mask)
        fused = torch.cat([text_reward, image_features], dim=1) 
        fused = torch.relu(self.fusion(fused)) 
        reward = self.fc(fused)
        return reward.squeeze(1)




reward_model = RewardModel()



image_encoder = torchvision.models.resnet18(pretrained=True)
multimodal_reward_model = MultimodalRewardModel(reward_model, image_encoder)





# Simple reward calculation function
def calculate_rewards(texts, image_features = None): 
    encoded_inputs = sft_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    
    with torch.no_grad():
        if image_features is None:
            rewards = reward_model(input_ids, attention_mask)
        else:
            rewards = multimodal_reward_model(input_ids, attention_mask, image_features)
    return rewards.cpu().numpy()

# sample 
generated_texts = ["This is a good text.", "This text is not so good."]
rewards = calculate_rewards(generated_texts)
print(rewards)


