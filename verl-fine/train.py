for batch in dataloader:
    prompts = batch['prompt']
    
    responses = policy_model.generate(prompts)
    
    rewards = reward_model(prompts, responses)
    values = value_model(prompts, responses)

    advantages = rewards - values.detach()

    policy_loss = compute_policy_loss(responses, advantages)
    value_loss = compute_value_loss(values, rewards)
    entropy_loss = compute_entropy_loss(policy_model)

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    total_loss.backward()
    optimizer.step()
