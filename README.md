# DualTAP: A Dual-Task Adversarial Protector for Mobile MLLM Agents

## 📖 Quick Start
```python
conda create -n DAP python=3.10
conda activate DAP
pip install -r requirements.txt
```


## 📚 DataSet-[PrivScreen](https://huggingface.co/datasets/fyzzzzzz/PrivScreen)

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

## 🧭 Training
```python
python train_map.py
```
change Surrogate model in config; attn_method = "contrast_pixel_grad"
The trained checkpoint is saved in /checkpoint_eot

## 🎯 Evaluation

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

## 📄 Citation

If you find this work useful, please kindly consider citing our paper:

```bibtex
@misc{zhang2025dualtapdualtaskadversarialprotector,
      title={DualTAP: A Dual-Task Adversarial Protector for Mobile MLLM Agents}, 
      author={Fuyao Zhang and Jiaming Zhang and Che Wang and Xiongtao Sun and Yurong Hao and Guowei Guan and Wenjie Li and Longtao Huang and Wei Yang Bryan Lim},
      year={2025},
      eprint={2511.13248},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2511.13248}, 
}
```