# Towards Robust and Distortion-free Semantic-level Watermarking With Multiple Channel Constraints
---

## Todo

1. [X] Upload the code.

## Get Start

- [Install](#install)
- [Dataset](#dataset-preparation)
- [Generation and Detection](#generation-and-detection)
- [Attacks](#attacks)

## Install

```shell
conda create -n pmark python=3.10
conda activate pmark
pip install -r requirements.txt
```

## Dataset Preparation

We have upload a demo of ``C4`` and ``BOOKSUM`` dataset in folder ``./data``.

## Generation and Detection

To generate watermarked text with **PMark**, run

```shell
bash scripts/gen.sh 
```

Also, to get help about arguments in ``pmark.py`` or ``detect.py``, run 
```shell
python pmark.py -h
python detect.py -h
```

## Attacks
We use standard attack methods implemented in MarkLLM(https://github.com/THU-BPM/MarkLLM), defined in ``attack_utils.py``. Therefore to conduct attack experiments, just put ``attack_utils.py`` and ``attack_pmark.py`` in the root dir of MarkLLM repo and run
```shell
bash scripts/attack.sh
```
