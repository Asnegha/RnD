from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import re
import pickle


def getTrainDataset():
    final_table_columns = [
        "dataset",
        "split",
        "passage",
        "question",
        "answer1",
        "answer2",
        "answer3",
        "answer4",
        "correct_answer",
        "correct_answer_num",
    ]
    df = pickle.load(open("belebele.pkl",'rb'))[final_table_columns]

    
    valds = df[df["split"]=="dev"].dropna()

    return trainds, valds

instr = "You are given a paragraph, followed by a question and 4 options 1,2,3 and 4. Your reply must be a single number 1, 2, 3 or 4 corresponding to the option you think answers the given question correctly."
def preprocess_function(df):
    print("***************************",type(df))
    custom_ds = pd.DataFrame()
    df["correct_answer_num"] = df["correct_answer_num"].apply(str)
    df["prompt"] = "<s>" + "[INST]" + instr  + df["passage"] + "\n" + df["question"]  +"\nOption 1\n"+ df["answer1"]+ "\nOption 2\n"+ df["answer2"]+ "\nOption 3\n"+ df["answer3"]+ "\nOption 4\n"+ df["answer4"]+ "[/INST]" + "The correct option number is: Option " + df["correct_answer_num"] + "</s>"
    custom_ds["prompt"] = df["prompt"]
    custom_ds.to_csv("data.csv")
    dataset = ds.dataset(pa.Table.from_pandas(custom_ds).to_batches())
    dataset = Dataset(pa.Table.from_pandas(custom_ds))
    return dataset



if __name__=='__main__':

    valds = getTrainDataset()
    trainds = pd.read_csv("train_10k.csv")
    train = preprocess_function(trainds) 
    val = preprocess_function(valds)

    base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    new_model = "fine_tuned_Mistral"

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.bos_token, tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
    evaluation_strategy = "epoch",
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    )

    # Setting sft parameters
    trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    max_seq_length=512
    )

    trainer.train()

    trainer.model.save_pretrained(new_model)
    model.config.use_cache = True
    model.eval()