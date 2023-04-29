---
language: en
datasets:
- multi-instruct
---

# Otter-9B

[Code](https://github.com/Luodian/PET-VLM) | [Demo](https://otter.cliangyu.com/)

Otter is an instruction-following large multi-model model built upon Open-Flamingo-9B. Through in-context instruction following, Otter is able to perform tasks more aligned with human preferences and more accurate.

## Model Details

Following the same setting as Flamingo, we freeze the pretrained vision encoder and language model, and only train Perceiver modules and cross-attention layers. We add one special token `<answer>` and resize the input and output embedding of the language model. This special token is used to separate the instruction and answer when calculating the causal loss ans used as a beginning token when generating the answer.

Our training data will be released soon.

## Uses

Otter-9B is intended to be used **for academic research purposes only.** Commercial use is prohibited, in line with LLaMA's non-commercial license.

### Bias, Risks, and Limitations

This model may generate inaccurate or offensive outputs, reflecting biases in its training data and pretrained priors.
