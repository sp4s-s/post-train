import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", streaming=True)

def preprocess_function(examples):
    return {
        "input_text_chosen": examples["chosen"],
        "input_text_rejected": examples["rejected"]
    }

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["input_text_chosen"],
        examples["input_text_rejected"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_train = tokenized_dataset["train"].map(tokenize_function, batched=True)
tokenized_test = tokenized_dataset["test"].map(tokenize_function, batched=True)

class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=1,  # Reduced for streaming
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=1000
)

trainer = Trainer(
    model=RewardModel(),
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)

trainer.train()