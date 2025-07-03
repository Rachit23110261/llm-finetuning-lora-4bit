# Fine-Tuning a Language Model using LoRA and 4-bit Quantization

## Overview

This repository contains the implementation of a project focused on fine-tuning a large language model (~1B parameters) using Low-Rank Adaptation (LoRA) and 4-bit quantization techniques. The objective is to perform efficient fine-tuning under limited GPU resources and evaluate the impact of fine-tuning through both quantitative metrics and qualitative analysis.

## Objectives

- Fine-tune a pre-trained language model using the Hugging Face Transformers library.
- Apply LoRA and 4-bit quantization for memory- and compute-efficient training.
- Evaluate model performance using both standard metrics and human evaluation.
- Train using a limited-resource environment such as Google Colab with a T4 GPU.

## Learning Outcomes

- Gained experience with Hugging Face Transformers and Datasets libraries.
- Implemented LoRA-based parameter-efficient fine-tuning with 4-bit quantized models.
- Carried out data preprocessing, training, and evaluation in a resource-constrained setup.
- Analyzed fine-tuned model outputs qualitatively and quantitatively.


## Model and Training Details

- **Pre-trained Model:** Example – `EleutherAI/gpt-neo-1.3B`
- **Quantization:** 4-bit loading using bitsandbytes
- **Adapter Technique:** Low-Rank Adaptation (LoRA) via Hugging Face PEFT
- **Training API:** Hugging Face Trainer

### Hyperparameters

| Parameter        | Value     |
|------------------|-----------|
| Epochs           | 3         |
| Batch Size       | 8         |
| Learning Rate    | 2e-4      |
| LoRA Rank        | 8         |
| LoRA Dropout     | 0.05      |
| Max Sequence Len | 256       |

## Evaluation

### Quantitative Evaluation

- Evaluated the base and fine-tuned models on a held-out validation set using language modeling metrics such as perplexity.

### Qualitative Evaluation

- Compared sample generations from both models based on fluency, coherence, and correctness.
- Conducted human evaluation by assigning a score (0 to 5) to each sample.

## Compute Resources

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- Techniques used for efficiency:
  - 4-bit quantization to reduce memory usage
  - LoRA to minimize the number of trainable parameters


