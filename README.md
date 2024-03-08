# Fine-Tune Large Languages Models and Evaluation
##  0 Environment Setup
- Train Evironment : NIVIDA A100
- Framework: Huggingface, Pytorch
- TRL: Transformer Reinforcement Learning [(TRL)](https://github.com/huggingface/trl)
- Accelerateor: [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main)
- Evaluate open LLMs on MT-Bench: MT-Bench is a Benchmark designed by LMSYS to test the conversation and instruction-following capabilities of large language models (LLMs). It evaluates LLMs through multi-turn conversations, focusing on their ability to engage in coherent, informative, and engaging exchanges. Since human evaluation is very expensive and time consuming, LMSYS uses GPT-4-Turbo to grade the model responses. MT-Bench is part of the [FastChat Repository](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md).

##  1 Fine-tune Google's GEMMA 7B based on databricks-dolly-15k dataset 
[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Gemma7B_Fine_Tuning.ipynb)


- When working with QLoRA, training focuses solely on adapter parameters, not the entire model. Consequently, only adapter weights are saved during training. To facilitate text generation inference, you can save the full model by merging the adapter weights with the main model weights using the merge_and_unload method. Subsequently, save the complete model using the save_pretrained method. This will create a standard model suitable for inference tasks.  

    - [Fine-tuned GEMMA 7B Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-chatml)  
    - [Fine-tuned GEMMA 7B Full Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-full-chatml)

## 2 Evaluate the Fine-tune models

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Evaluation.ipynb)

## 3 Fine-tune CodeLLAMA for Text to SQL 

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Fine-Tuning%20codellama.ipynb)

[Fine-tuned CodeLLAMA for Text-to-SQL Model](https://huggingface.co/jinhybr/code-llama-7b-text-to-sql)

[Fine-tuned Mistral-7B for Text-to-SQL Model](https://huggingface.co/jinhybr/Mistral-7B-v0.1-text-to-sql)
##  4 Deployment Demo

[Inference usage Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/deploy.ipynb)

[Gradio App](https://github.com/Pyligent/finetune-LLM/blob/main/app.py)

[HuggingFace Space](https://huggingface.co/spaces/jinhybr/finetune-code-llama-7b-Text-to-SQL-Demo)
 
