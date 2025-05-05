from datasets import load_dataset
hh_rlhf = load_dataset("Anthropic/hh-rlhf")
train_dataset = hh_rlhf["train"]