# Dual-Attribute Adversarial Protection Against Privacy Leakage in Mobile MLLM Agents

## Quick Start
```python
conda create -n DAP python=3.10
conda activate DAP
pip install -r requirements.txt
```


## Dataset
```python
/data
    --/{app}
        --/images
            --{app}_01.png
            ...
        --normal_qa.json
        --privacy_qa.json
    --/{app2}
    ...
```

## Training
```python
python train_map.py
```
change Surrogate model in config; attn_method = "contrast_pixel_grad"
The trained checkpoint is saved in /checkpoint_eot

## Evaluation

### Eval with Generator
```python
python eval.py \
  --checkpoint {path}.pth \
  --output ./eval_results/{path}/eval.json \
  --llm-model gpt-5-mini
```
### Eval without Generator
```python
python eval_original.py \
    --output ./eval_results/{path}/original.json \
    --llm-model gpt-5-mini
```
### Using API (GPT, Gemini, Openrouter)
```python
python eval.py \
  --checkpoint {path}.pth \
  --use-api --api-type openai \
  --api-model gpt-5 \
  --output ./eval_results/gpt/eval_our.json \
  --llm-model gpt-5-mini
```