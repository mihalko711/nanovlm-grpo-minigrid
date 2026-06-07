# nanoVLM-GRPO-MiniGrid

Пайплайн выравнивания (alignment) мультимодальной модели nanoVLM (460M) для управления агентом в среде MiniGrid-Empty-16×16-v0.

Сравниваются три подхода: **SFT**, **GRPO-Action**, **GRPO-Reasoning**.  
Модель работает в режиме VQA: принимает карту среды + текстовый промпт, возвращает следующее действие.

## Установка

```bash
git clone --recurse-submodules https://github.com/...
pip install gymnasium minigrid pillow torch transformers datasets einops
```

Если уже склонировано без субмодуля:

```bash
git submodule update --init --recursive
```

## Структура

```
env_utils.py        — Обёртка MiniGrid (EmptyEnv, agent_start_pos=None, рендер 7×7 POV)
expert.py           — Реактивный эксперт (POV-based) + генератор датасета
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
python expert.py                  # 20 экспертных эпизодов → data/sft_dataset.jsonl (по умолч.)
python expert.py crop 100         # 100 эпизодов в crop-режиме
jupyter notebook sft_train.ipynb  # SFT проектора
python evaluate.py --checkpoint checkpoints/sft_model.pt  # замер success rate
```

## Модель

База: [`lusxvr/nanoVLM-460M-8k`](https://huggingface.co/lusxvr/nanoVLM-460M-8k)  
- Vision: SigLIP2-B/16-512 (85M)
- Language: SmolLM2-360M-Instruct (360M)
- Projector: pixel shuffle 4× + Linear

Обучение: только ModalityProjector, vision encoder и decoder заморожены.
