# Fine-Tune Large Languages Models and Evaluation
## Key Summary
###  1 Environment Setup
- Train Evironment : NIVIDA A100
- Framework: Huggingface, Pytorch
- TRL: Transformer Reinforcement Learning [(TRL)](https://github.com/huggingface/trl)
- Accelerateor: [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main)
- Evaluate open LLMs on MT-Bench: [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) is a Benchmark designed by LMSYS to test the conversation and instruction-following capabilities of large language models (LLMs). It evaluates LLMs through multi-turn conversations, focusing on their ability to engage in coherent, informative, and engaging exchanges. Since human evaluation is very expensive and time consuming, LMSYS uses GPT-4-Turbo to grade the model responses. MT-Bench is part of the [FastChat Repository](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md).


###  2 Dataset
Important Datasets for fine-tuning LLMs:
- Using existing open-source datasets, e.g., [Spider](https://huggingface.co/datasets/spider), [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- Using LLMs to create synthetically datasets, e.g., [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca),[Ultrafeedback](https://www.notion.so/9de9ac96f0f94aa5aed96361a26e8bf0?pvs=21)
- Using Humans to create datasets, e.g., [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k),[HH](https://www.notion.so/SageMaker-bi-weekly-sync-0be2e6ba876a4599b4c0da2681dfb78f?pvs=21)
- Using a combination of the above methods, e.g., [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca),[Orca DPO](https://huggingface.co/datasets/Intel/orca_dpo_pairs)

The choice of dataset and format depends on the specific scenario and use case. But, preference datasets can inherit biases from the humans or AI that created them. To ensure broader applicability and fairness, incorporate diverse feedback when constructing these datasets.


### 3 TRL and SFTTrainer for Fine-Tuning

SFTTrainer builds upon the robust foundation of the Trainer class from Transformers, offering all the essential features: logging, evaluation, and checkpointing. But SFTTrainer goes a step further by adding a suite of features that streamline the fine-tuning process:

- Dataset Versatility: SFTTrainer seamlessly handles conversational and instruction formats, ensuring for optimal training.
- Completion Focus: SFTTrainer prioritizes completions for training, ignoring prompts and maximizing efficiency.
- Packing datasets: SFTTrainer packs a punch by optimizing your datasets for faster, more efficient training runs.
- Fine-Tuning Finesse: Unleash the power of PEFT (Parameter-Efficient Fine-Tuning) techniques like Q-LoRA. It reduces memory usage during fine-tuning without sacrificing performance through a process called quantization.
- Conversational Readiness: SFTTrainer equips model and tokenizer for conversational fine-tuning by equipping them with essential special tokens.

###  4 Evaluation
[MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) offers two evaluation strategies:

- Single-Answer Grading: LLMs directly grade their own answers on a 10-point scale.
- Pair-Wise Comparison: LLMs compare two responses and determine which one is better, resulting in a win rate.

For our evaluation, we will leverage the pair-wise comparison method to compare fine-tuned gemma-7b-Dolly15k-chatml with Mistral-7B-Instruct-v0.2 model.
MT-Bench currently only supports OpenAI or Anthropic as the judge, in this case , so we will leverage the [FastChat Repository](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) and incorporate reference answers from GPT-4 Turbo (gpt-4-1106-preview). This allows us to maintain a high-quality evaluation process without incurring significant costs. 

- Generate Responses using gemma-7b-Dolly15k-chatml and Mistral-7B-Instruct-v0.2
- Evaluate the responses using pair-wise comparison and GPT-4-Turbo as Judge
- - Plot and compare the results

## Notebooks
###  1 Fine-tune Google's GEMMA 7B based on databricks-dolly-15k dataset 
[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Gemma7B_Fine_Tuning.ipynb)


- When working with QLoRA, training focuses solely on adapter parameters, not the entire model. Consequently, only adapter weights are saved during training. To facilitate text generation inference, you can save the full model by merging the adapter weights with the main model weights using the merge_and_unload method. Subsequently, save the complete model using the save_pretrained method. This will create a standard model suitable for inference tasks.  

    - [Fine-tuned GEMMA 7B Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-chatml)  
    - [Fine-tuned GEMMA 7B Full Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-full-chatml)

### 2 Evaluate the Fine-tune models

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Evaluation.ipynb)

### 3 Fine-tune CodeLLAMA for Text to SQL 

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Fine-Tuning%20codellama.ipynb)

[Fine-tuned CodeLLAMA for Text-to-SQL Model](https://huggingface.co/jinhybr/code-llama-7b-text-to-sql)

[Fine-tuned Mistral-7B for Text-to-SQL Model](https://huggingface.co/jinhybr/Mistral-7B-v0.1-text-to-sql)
###  4 Deployment Demo

[Inference usage Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/deploy.ipynb)

[Gradio App](https://github.com/Pyligent/finetune-LLM/blob/main/app.py)

Simple Demo on Huggingface Space
[HuggingFace Space](https://huggingface.co/spaces/jinhybr/finetune-code-llama-7b-Text-to-SQL-Demo)
 
