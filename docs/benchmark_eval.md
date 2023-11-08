# Welcome to the benchmark evaluation page!

The evaluation pipeline is designed to be one-clickable and easy to use. However, you may encounter some problems when running the models (e.g. LLaVA, LLaMA-Adapter) that require you to clone their repo to local path. Please feel free to contact us if you have any questions.

We support the following benchmarks:
- MagnifierBench
- MMBench
- MM-VET
- MathVista
- POPE
- MME
- SicenceQA
- SeedBench

And following models:
- LLaVA
- Fuyu
- OtterHD
- Otter-Image
- Otter-Video
- Idefics
- LLaMA-Adapter
- Qwen-VL

many more, see `/pipeline/benchmarks/models`

Create a yaml file `benchmark.yaml` with below content:
```yaml
datasets:
  - name: magnifierbench
    split: test
    data_path: Otter-AI/MagnifierBench
    prompt: Answer with the option letter from the given choices directly.
    api_key: [You GPT-4 API]
  - name: mme
    split: test
  - name: pope
    split: test
    default_output_path: ./logs
  - name: mmvet
    split: test
    api_key: [You GPT-4 API]
    gpt_model: gpt-4-0613
  - name: mathvista
    split: test
    api_key: [You GPT-4 API]
    gpt_model: gpt-4-0613
  - name: mmbench
    split: test
models:
  - name: fuyu
    model_path: adept/fuyu-8b
```

Then run

```python
python -m pipeline.benchmarks.evaluate --confg benchmark.yaml
```