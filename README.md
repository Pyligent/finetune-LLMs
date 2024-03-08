# Fine-Tune Large Languages Models and Evaluation
##  1 Fine-tune Google's GEMMA 7B based on databricks-dolly-15k dataset 
[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Gemma7B_Fine_Tuning.ipynb)


- When working with QLoRA, training focuses solely on adapter parameters, not the entire model. Consequently, only adapter weights are saved during training. To facilitate text generation inference, you can save the full model by merging the adapter weights with the main model weights using the merge_and_unload method. Subsequently, save the complete model using the save_pretrained method. This will create a standard model suitable for inference tasks.
[Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-chatml)
[Full Model](https://huggingface.co/jinhybr/gemma-7b-Dolly15k-full-chatml)

## 2 Evaluate the Fine-tune models

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Evaluation.ipynb)

## 3 Fine-tune CodeLLAMA for Text to SQL 

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/Fine-Tuning%20codellama.ipynb)

[Fine-tuned CodeLLAMA for Text-to-SQL Model](https://huggingface.co/jinhybr/code-llama-7b-text-to-sql)

[Fine-tuned Mistral-7B for Text-to-SQL Model](https://huggingface.co/jinhybr/Mistral-7B-v0.1-text-to-sql)
##  4 Deployment Demo

[Notebook](https://github.com/Pyligent/finetune-LLM/blob/main/deploy.ipynb)

[HuggingFace Space](https://huggingface.co/spaces/jinhybr/finetune-code-llama-7b-Text-to-SQL-Demo)
 
