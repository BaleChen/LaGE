# Finetuning Large Language Models with RLHF and Guided Exploration
Project done at NYU Shanghai Frontiers Science Center of Artificial Intelligence and Deep Learning, supervised by [Prof. Wilson Tam](https://shanghai.nyu.edu/academics/faculty/directory/yik-cheung-wilson-tam).

This is an implementation of Reinforcement Learning from Human Feedback using HuggingFace TRL. I used this codebase to train Llama-7b and Llama-13b on Anthrophic's [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset and Jeremy Scheurer et al.'s [summarization](https://huggingface.co/datasets/JeremyAlain/SLF5K) dataset. 

## Guided Exploration

In reinforcement learning, the exploration-exploitation tradeoff is one of the more essential topics. In RLHF, the exploration is done by random sampling during decoding time. We found that this potentially caused the instability of RLHF training. We probed into using another language model to guide the target policy model's generation for faster and stabler convergence during the PPO stage.
