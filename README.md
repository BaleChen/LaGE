# Finetuning Large Language Models with RLHF and Guided Exploration
Project done at NYU Shanghai Frontiers Science Center of Artificial Intelligence and Deep Learning, supervised by [Prof. Wilson Tam](https://shanghai.nyu.edu/academics/faculty/directory/yik-cheung-wilson-tam).

This is an implementation of Reinforcement Learning from Human Feedback using HuggingFace TRL. I used this codebase to train Llama-7b and Llama-13b on Anthrophic's [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset and Jeremy Scheurer et al.'s [summarization](https://huggingface.co/datasets/JeremyAlain/SLF5K) dataset. 

## Guided Exploration

In reinforcement learning, the exploration-exploitation tradeoff is one of the more essential topics. In RLHF, the exploration is done by random sampling during decoding time. We found that this potentially caused the instability of RLHF training. We probed into using another language model to guide the target policy model's generation for faster and stabler convergence during the PPO stage.

Although this is not a successful approach, here we open our scripts for the research community and list a few challenging points that we encountered:
* One common behavior is that the target LM generates less and less during the PPO stage. One possible remedy is to add a verbosity reward to award longer answers, but we didn't have time to try it out.
* Negative KL-divergence is a common but deceptive behavior of PPO training. When we have negative KL, the training reward seems to increase, but the model is not learning. We can follow TRL's fix [here](https://huggingface.co/docs/trl/how_to_train#what-is-the-concern-with-negative-kl-divergence).
* TRL is an actively maintained library for RL with Transformer models. Be sure to regularly check for updates.
