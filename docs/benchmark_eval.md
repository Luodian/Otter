Welcome to the benchmark evaluation page!

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
models:
  - name: fuyu
    model_path: adept/fuyu-8b
```

Then run

```python
python -m pipeline.benchmarks.evaluate --confg benchmark.yaml
```