import os
import torch
import torch.nn.functional as F
import requests
from megatron import get_args, initialize_megatron, mpu
from megatron.model import GPTModel
from megatron.tokenizer.tokenizer import build_tokenizer
from torch.distributed import all_reduce
from common import KLScheduler, RewardNormalizer, EarlyStopping, compute_gae

def vllm_generate(prompt, servers, max_tokens=128):
    server = servers[torch.randint(len(servers), (), dtype=torch.long).item()]
    resp = requests.post(
        f"{server}/generate",
        json={"prompt": prompt, "n": 1, "temperature": 0.8, "max_tokens": max_tokens},
        timeout=60,
    )
    return resp.json()["text"][0]

def verl_step(args, policy, value_head, optimizer, tokenizer, servers, kl_scheduler, normalizer, step):
    prompt = args.prompts[step % len(args.prompts)]
    ids = tokenizer.tokenize(prompt)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(ids)], device="cuda")
    
    with torch.no_grad():
        old_logits = policy(input_ids)[0][:, -1, :]
        old_log_probs = torch.log_softmax(old_logits, dim=-1)
    
    response = vllm_generate(prompt, servers, max_tokens=args.max_new_tokens)
    raw_reward = len(response.split()) / 10.0
    shaped_reward = normalizer.normalize(raw_reward)
    
    resp_ids = tokenizer.tokenize(response)
    response_ids = torch.tensor([tokenizer.convert_tokens_to_ids(resp_ids)], device="cuda")
    gen_ids = torch.cat([input_ids, response_ids], dim=1)
    
    output = policy(gen_ids)[0]
    logits = output[:, :-1, :]
    targets = gen_ids[:, 1:]
    new_log_probs = torch.log_softmax(logits, dim=-1)
    
    selected_new = new_log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
    old_selected = old_log_probs.gather(1, input_ids[:, -1].unsqueeze(-1)).squeeze(-1)
    
    kl_coeff = kl_scheduler.get_kl(step)
    kl_loss = (selected_new - old_selected).mean() * kl_coeff
    
    last_hidden = output[:, -1, :]
    values = value_head(last_hidden).squeeze()
    rewards = shaped_reward * torch.ones_like(values)
    advantages = compute_gae(rewards.tolist(), values.tolist(), gamma=args.gamma, lam=args.lam)
    
    ratio = torch.exp(selected_new - old_selected)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    value_loss = F.mse_loss(values, rewards) * args.value_coef
    entropy_loss = -new_log_probs.mean() * args.entropy_coef
    total_loss = policy_loss + value_loss + entropy_loss + kl_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    all_reduce(total_loss, group=mpu.get_data_parallel_group())
    optimizer.step()
    
    return total_loss.item()

def main():
    initialize_megatron(extra_args_provider=None)
    args = get_args()
    
    args.prompts = ["Explain gravity.", "Describe the ocean.", "Tell me about AI."]
    args.max_new_tokens = getattr(args, 'max_new_tokens', 64)
    args.gamma = 0.99
    args.lam = 0.95
    args.clip = 0.2
    args.value_coef = 0.5
    args.entropy_coef = 0.01
    args.eval_interval = 1000
    args.checkpoint_interval = 500
    
    servers = os.environ.get("VLLM_SERVERS", "http://localhost:8000").split(",")
    tokenizer = build_tokenizer(args)
    policy = GPTModel(num_tokentypes=0, parallel_output=True).cuda()
    value_head = torch.nn.Linear(args.hidden_size, 1).cuda()
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_head.parameters()), lr=2e-5)
    kl_scheduler = KLScheduler(init_kl=0.1, final_kl=0.02, total_steps=10000)
    normalizer = RewardNormalizer(alpha=0.99)
    early_stopper = EarlyStopping(patience=3, min_delta=0.01)
    
    step = 0
    while True:
        loss = verl_step(args, policy, value_head, optimizer, tokenizer, servers, kl_scheduler, normalizer, step)
        
        if mpu.get_data_parallel_rank() == 0 and step % 100 == 0:
            print(f"Step {step} Loss: {loss:.4f}")
        
        if step > 0 and step % args.eval_interval == 0:
            scores = []
            for p in args.prompts:
                r = vllm_generate(p, servers, max_tokens=args.max_new_tokens)
                scores.append(len(r.split()) / 10.0)
            avg = sum(scores) / len(scores)
            if early_stopper.step(avg):
                if mpu.get_data_parallel_rank() == 0:
                    print("Early stopping.")
                break
        
        if step % args.checkpoint_interval == 0 and mpu.get_data_parallel_rank() == 0:
            ckpt = {
                'model': policy.state_dict(),
                'value_head': value_head.state_dict(),
                'opt': optimizer.state_dict(),
                'step': step
            }
            torch.save(ckpt, f"{args.save}/ckpt_{step}.pt")
        
        step += 1

if __name__ == '__main__':
    main()