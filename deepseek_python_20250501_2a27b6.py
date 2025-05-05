import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf")

def preprocess_function(examples):
    return {
        "input_text_chosen": examples["chosen"],
        "input_text_rejected": examples["rejected"]
    }

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized_chosen = tokenizer(examples["input_text_chosen"], padding="max_length", truncation=True)
    tokenized_rejected = tokenizer(examples["input_text_rejected"], padding="max_length", truncation=True)
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"]
    }

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

class RewardModel(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1, problem_type="regression")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)

def preference_loss(chosen_rewards, rejected_rewards):
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss

class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_rewards = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])
        rejected_rewards = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])
        loss = preference_loss(chosen_rewards, rejected_rewards)
        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss

training_args = TrainingArguments(
    output_dir="./reward_model_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=1_000,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,
    gradient_accumulation_steps=4,
    report_to="none"
)

model = RewardModel(base_model_name=model_name)
trainer = RewardModelTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

trainer.train()
model.save_pretrained("./trained_reward_model")
tokenizer.save_pretrained("./trained_reward_model")