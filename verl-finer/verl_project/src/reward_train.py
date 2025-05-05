import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def train_reward_model(model_name="gpt2", dataset_name="Anthropic/hh-rlhf", output_dir="checkpoints/rm"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    ds = load_dataset(dataset_name, split="train")
    for epoch in range(3):
        for batch in ds.shuffle().select(range(1000)):
            text = batch.get("chosen", batch.get("text", ""))
            reward = float(batch.get("score", 0.0))
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            labels = torch.tensor([[reward]], device="cuda")
            outputs = model(**inputs)
            loss = F.mse_loss(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_reward_model()
