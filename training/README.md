# Training code

This folder contains the scripts used to run the beta sweeps.  Each script
accepts at least:

```bash
--beta <float> --batch_size <int> --seed <int>
```

The common beta grid is stored in `beta_grid.txt`.  Balanced Adam is imposed by
setting `beta_1 = beta_2 = beta` inside the optimizer construction.

## Seed policy

Every experiment and every beta in the grid was run with seed 1.  Seeds 2 and 3
were added only for a small near-optimal beta window identified from the seed-1
sweep.  Therefore, averages over seeds are available only in those local
windows.  T5-small uses seed 1 only in the final analysis because the extra T5
seeds were treated as unreliable.

## Dataset locations

By default, scripts look for datasets under:

```bash
DATASETS_ROOT=/scratch1/hernanal/datasets
```

Set `DATASETS_ROOT` or pass `--data_root` when supported if your data live
elsewhere.

## Development experiments

ResNet50 on Food-101 uses `experiments/resnet50_food101.py`, batch size 128,
50 epochs, random initialization, AdamW with `beta_1=beta_2=beta`, weight decay
0.01 on decay parameters, cosine schedule with warmup, label smoothing 0.1 for
training loss, and standard ImageNet normalization.  Evaluation is performed
once per epoch.

ResNet50 on ImageNet100 uses `experiments/resnet50_imagenet100.py`, batch size
128, 80 epochs, random initialization, AdamW with `beta_1=beta_2=beta`, the
same ResNet training recipe as Food-101, and evaluation once per epoch.

ViT-B/16 on CIFAR-100 uses `experiments/vitb16_cifar100.py`, batch size 128,
50 epochs, random initialization, AdamW with weight decay 0.1, cosine schedule
with 1000 warmup steps, gradient clipping 1.0, label smoothing 0.1, RandAugment,
RandomErasing, and ImageNet normalization.

ViT-B/16 on TinyImageNet uses `experiments/vitb16_tinyimagenet.py`, batch size
128, 50 epochs, random initialization, AdamW with the same ViT recipe as
CIFAR-100, and evaluation once per epoch.

NanoGPT on WikiText-103 uses `experiments/nanogpt_wikitext.py`, a randomly
initialized GPT-style model with context length 256, physical batch size 8,
gradient accumulation to 262144 tokens per update, AdamW with
`beta_1=beta_2=beta`, weight decay 0.01, cosine learning-rate decay, warmup
2000 steps, gradient clipping 1.0, 10000 training steps, and evaluation every
500 steps.

NanoGPT on OpenWebText uses `experiments/nanogpt_openwebtext.py`, the same
NanoGPT architecture and optimizer protocol as WikiText-103, using packed GPT-2
BPE tokens from OpenWebText.

Llama60M on C4 uses `experiments/llama60m_c4.py`, a randomly initialized
Llama60M configuration, tokenizer `t5-base`, max sequence length 256, physical
batch size 64, gradient accumulation to total batch size 512, Adam with
`beta_1=beta_2=beta`, weight decay 1e-5, cosine schedule with 1000 warmup
steps, 10000 training steps, and evaluation every 1000 steps.

Llama60M on SlimPajama-6B uses `experiments/llama60m_slimpajama6b.py`, the same
Llama60M architecture and optimizer protocol as C4, using SlimPajama-6B.

## Held-out experiments

T5-small on BookCorpus uses `experiments/t5small_bookcorpus.py`, pretrained
`t5-small`, span-corruption denoising, max input length 256, physical batch size
8, gradient accumulation to total batch size 64, AdamW with
`beta_1=beta_2=beta`, weight decay 0.01, learning rate 1e-4, cosine schedule,
gradient clipping 1.0, 10000 steps, and evaluation every 1000 steps.

Swin-T on Caltech-256 uses `experiments/swin_t_caltech256.py`, random
initialization, batch size 64, 60 epochs, AdamW with weight decay 0.05, cosine
schedule, gradient clipping 1.0, label smoothing 0.1, RandAugment,
RandomErasing, and evaluation once per epoch.

EfficientNet-B0 on Stanford Cars uses
`experiments/efficientnetb0_stanfordcars.py`, random initialization, batch size
64, 100 epochs, Adam with `beta_1=beta_2=beta`, weight decay 5e-5, cosine
schedule, gradient clipping 1.0, label smoothing 0.05, RandAugment, mild
RandomErasing, and evaluation once per epoch.

## Example command

```bash
python training/experiments/vitb16_cifar100.py \
  --beta 0.94377 \
  --batch_size 128 \
  --seed 1 \
  --data_root /path/to/cifar100
```
