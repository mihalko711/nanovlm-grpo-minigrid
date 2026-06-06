# nanoVLM-GRPO-MiniGrid

Пайплайн выравнивания (alignment) мультимодальной модели nanoVLM (460M) для управления агентом в среде MiniGrid-Empty-16×16-v0.

Сравниваются три подхода: **SFT**, **GRPO-Action**, **GRPO-Reasoning**.  
Модель работает в режиме VQA: принимает карту среды + текстовый промпт, возвращает следующее действие.

## Структура

```
env_utils.py        — Обёртка MiniGrid (16×16, рандомизация позиций, рендер 512×512)
expert.py           — BFS-солвер кратчайшего пути + генератор экспертного датасета
sft_train.ipynb     — SFT-обучение: заморозка vision/LM, тренировка ModalityProjector
evaluate.py         — Оценка агента: 100 эпизодов, success rate vs BFS-оракул
nanoVLM/            — Склонированный huggingface/nanoVLM (зависимость)
KAGGLE.md           — Инструкция по запуску на Kaggle
```

## Формат данных

```json
{"images": ["data/images/ep_001_step_00.png"], "texts": [{"user": "What is the next action...", "assistant": "move forward"}]}
```

Совместим с нативным `VQADataset` из nanoVLM.

## Пайплайн

```bash
python expert.py                  # 1000 экспертных эпизодов → data/sft_dataset.jsonl
jupyter notebook sft_train.ipynb  # SFT проектора
python evaluate.py --checkpoint checkpoints/sft_model.pt  # замер success rate
```

## Модель

База: [`lusxvr/nanoVLM-460M-8k`](https://huggingface.co/lusxvr/nanoVLM-460M-8k)  
- Vision: SigLIP2-B/16-512 (85M)
- Language: SmolLM2-360M-Instruct (360M)
- Projector: pixel shuffle 4× + Linear

Обучение: только ModalityProjector, vision encoder и decoder заморожены.
