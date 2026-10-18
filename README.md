# Exemplar-Based Continual Learning for Image Classification

Class-incremental learning experiments with [Avalanche](https://avalanche.continualai.org/) and PyTorch, comparing a **Naive fine-tuning baseline** with **iCaRL** (exemplar replay + distillation + nearest-mean-of-exemplars) on **Split CIFAR-100** and **CORe50**, with a study of the effect of memory size. A **MIR** (Maximally Interfered Retrieval) study is included in the report.

## Motivation

Deep networks trained sequentially on new classes suffer from **catastrophic forgetting**. This project studies the stability-plasticity trade-off in the **Class-Incremental Learning (CIL)** scenario: each experience brings new classes, and the model must classify among all classes seen so far with a single head.

## Methods

| Method | Type | Notes |
|---|---|---|
| Naive | Baseline | Sequential fine-tuning, no replay, no regularisation |
| iCaRL | Exemplar-based | Herding exemplar selection, knowledge distillation, NME classifier |
| MIR | Replay-based | Replays the memory samples most interfered by the incoming update (see report) |

## Benchmarks and setup

| | Split CIFAR-100 | CORe50 (NC scenario) |
|---|---|---|
| Experiences | 10 (10 classes each) | 9 |
| Classes | 100 | 50 |
| Backbone | ResNet-18 | ResNet-18 |

Common settings: SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch size 128, 50 epochs per experience, seed 42. iCaRL uses an ImageNet-pretrained ResNet-18 (CIFAR variant: 3x3 stem, no max-pool for CIFAR-100), a `MultiStepLR` schedule (milestones 30 and 40, gamma 0.1) and memory sizes **500 / 1000 / 2000 / 5000**.

## Metrics

- **AA**: accuracy after the last experience
- **AIA**: average of the accuracy curve over experiences
- **FM**: forgetting measure (Avalanche `StreamForgetting`)

> **Evaluation protocol note.** On CIFAR-100, the "seen-only" scripts (`*_tests2`) average accuracy over the experiences seen so far. On CORe50 the Avalanche test stream is a single experience containing all 50 classes, so stream accuracy is computed over seen and unseen classes alike and rises as more classes are learned. For the same reason, FM is not directly meaningful on CORe50. Do not compare CIFAR-100 and CORe50 curves without keeping this in mind.

## Main results

Naive baseline (single run, seed 42):

| Benchmark | AA | AIA | FM |
|---|---|---|---|
| Split CIFAR-100 | 0.0749 | 0.2077 | 0.7163 |
| CORe50 | 0.0814 | 0.0800 | 0.0573 |

iCaRL (see the report for the full curves):

- **CIFAR-100**: final accuracy around 26-27%, AIA around 45-47%, FM 0.35-0.41. Larger memory only helps modestly because there are few exemplars per class.
- **CORe50**: final accuracy increases with memory, from about 32% (500) to 45.5% (5000), versus about 8% for Naive.

Naive accuracy on CIFAR-100 drops from 67.3% after the first experience to 7.5% after the last, which illustrates catastrophic forgetting. iCaRL clearly mitigates it on both benchmarks.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── experiments/
│   ├── m_baseline_cifar100.py          # Naive on Split CIFAR-100 (seen-only accuracy curve)
│   ├── m_baseline_core50.py            # Naive on CORe50 (full-stream accuracy)
│   ├── m_iCaRL_cifar100_tests.py       # iCaRL CIFAR-100: full-stream accuracy + accuracy matrix
│   ├── m_iCaRL_cifar100_tests2.py      # iCaRL CIFAR-100: seen-only accuracy curve
│   ├── m_iCaRL_core50_tests1.py        # iCaRL CORe50: full-stream accuracy + accuracy matrix
│   └── m_iCaRL_core50_tests2.py        # iCaRL CORe50: seen-only variant
├── plots/
│   ├── m_trace.py                      # Naive vs iCaRL, all memory sizes (CIFAR-100)
│   ├── m_trace2.py                     # iCaRL CIFAR-100 vs CORe50 (memory 5000)
│   ├── m_trace_curve_icarl.py          # iCaRL curves with value labels
│   └── m_trace_curve_naive.py          # Naive curve with value labels
└── results/                            # CSV logs produced by the experiments
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended: each run trains 50 epochs per experience.

## Usage

Datasets are downloaded automatically by Avalanche on first use (CIFAR-100 is small; CORe50 is a large download).

```bash
# Naive baselines
python experiments/m_baseline_cifar100.py
python experiments/m_baseline_core50.py

# iCaRL with a memory sweep (500, 1000, 2000, 5000)
python experiments/m_iCaRL_cifar100_tests2.py
python experiments/m_iCaRL_core50_tests1.py
```

Each script appends its results to CSV files (`*_log*.csv`, `*_curve*.csv`, `*_matrix*.csv`) in the working directory. Then plot:

```bash
python plots/m_trace.py
```

The plotting scripts read hard-coded CSV filenames defined at the top of each file. Edit them to match the CSVs you generated.

## Known limitations

- Single seed (42); no confidence intervals.
- The Naive baseline is trained from scratch, whereas iCaRL starts from an ImageNet-pretrained backbone, so the gap between them also reflects the backbone.
- CIFAR-100 and CORe50 curves use different accuracy definitions (see the protocol note above).
- Memory usage and training time were not measured.

## References

1. Rebuffi et al., *iCaRL: Incremental Classifier and Representation Learning*, CVPR 2017.
2. Aljundi et al., *Online Continual Learning with Maximal Interfered Retrieval*, NeurIPS 2019.
3. Lomonaco and Maltoni, *CORe50: a New Dataset and Benchmark for Continuous Object Recognition*.
4. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
5. Wang et al., *A Comprehensive Survey of Continual Learning: Theory, Method and Application*.
6. ContinualAI, [Avalanche](https://avalanche.continualai.org/).

## License

Released under the MIT License.
