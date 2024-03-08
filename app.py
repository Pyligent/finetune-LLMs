mport gradio as gr
from transformers import pipeline

# Load the pre-trained model from Hugging Face

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

peft_model_id = "jinhybr/code-llama-7b-text-to-sql"
# peft_model_id = args.output_dir

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)



def text_to_sql(text):
    # Load Model with PEFT adapter
    
    # Define schema and user question
    #schema = "CREATE TABLE table_17429402_7 (school VARCHAR, last_occ_championship VARCHAR)"
    schema = 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_17429402_7 (school VARCHAR, last_occ_championship VARCHAR)'
    user_question = text
    #user_question =  'How many schools won their last occ championship in 2006?'

    # Combine schema and user question
    combined_json_data = [
        {'content': schema, 'role': 'system'},
        {'content': user_question, 'role': 'user'}
    ]

    # Generate SQL query
    prompt = pipe.tokenizer.apply_chat_template(combined_json_data, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    sql_query = outputs[0]['generated_text'][len(prompt):].strip()

    return sql_query

# Create Gradio Interface
iface = gr.Interface(
    fn=text_to_sql,
    #inputs=gr.inputs.Textbox(lines=7, label="User Question"),
    #inputs=gr.inputs.Textbox(lines=7, label="User Question"),
    inputs = ['text'],
    outputs=['text'],
    theme="soft",
    examples=['How many schools won their last occ championship in 2006?'],
    cache_examples=True,
    title="Finetuned code-llama-7b for Text-to-SQL Demo",
    description="Translate text to SQL query based on the provided schema.CREATE TABLE table_17429402_7 (school VARCHAR, last_occ_championship VARCHAR)"
)
iface.launch()
