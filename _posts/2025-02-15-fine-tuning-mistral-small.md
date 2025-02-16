---
layout: post
author: initialxy
title: "Fine-tuning LLM Mistral Small"
description: "fine-tuning Mistral Small and others on AMD consumer GPU"
category: "Lesson"
tags: [MachineLearning, LLM, LoRA, Mistral-Small]
---
{% include JB/setup %}

I want to write a small follow-up to my [last post](/lesson/2025/01/31/fine-tuning-llm-on-amd-gpu), which used [Phi-4](https://huggingface.co/microsoft/phi-4) as the base model. I decided to switch to the newly released [Mistral-Small](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) instead because I quite like it. While working on it, I discovered some more quirks that I believe are worth discussing. I know in my last post, I mentioned that I would discuss how to use a fine-tuned model with RAG. That will be a follow-up to this post, and it will use the Mistral-Small-based fine-tuning discussed in this post. So please stay tuned.

<!--more-->

## Why Mistral-Small?
More specifically why did I choose Phi-4? Simply because it was a relatively small model that could fit in my VRAM and performed quite well. For the same reason, I felt the newly released Mistral-Small also fit these criteria and performed better than Phi-4. So I wanted to see how much effort it would take to switch to a different base model. Specifically, one that's brand new. Surprisingly, it wasn't as straightforward as I thought it would be.

## Changes in Training Script
Let's get straight to business. Here is my updated training script

`train_mistral.py`
```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from unsloth.save import unsloth_save_pretrained_merged

base_model_name = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"

# Load base model to GPU memory.
device = "cuda:0"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
        base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("json", data_files="data.jsonl", split="train")

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

peft_config = LoraConfig(
    lora_alpha = 8,
    lora_dropout = 0.1,
    r = 64,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

trainer = SFTTrainer(
    model = base_model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    peft_config=peft_config,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        max_steps = -1,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3820,
        output_dir = "outputs",
        report_to = "none"
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "[INST]",
    response_part = "[/INST]",
)

trainer_stats = trainer.train()

unsloth_save_pretrained_merged(trainer.model, "mistral_frieren", tokenizer, save_method="merged_16bit")
```

Thankfully, unsloth already published [bnb 4bit version of Mistral-Small-24B-Instruct-2501](https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit) on huggingface, so we could just pull from there. However looking at their [Mistral notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb), it doesn't look entirely correct. Perhaps it's outdated. First of all, `chat_template` should use `mistral` instead of `chatml`, and second, their notebook does not use `train_on_responses_only`. The first part is relatively intuitive, but how do we know what `instruction_part` and `response_part` should be used for `train_on_responses_only`? To get the correct value for that, you need to look at [Mistral-Small's template](https://ollama.com/library/mistral-small:24b/blobs/6db27cd4e277). `instruction_part` and `response_part` are essentially separators that indicate instruction and response. Looking at the actual template could be a bit confusing, it may be easier to go into Python REPL, copy and paste the first part of this script, and just print out some samples from the training dataset. Eg.

```python
>>> dataset['text'][0]
"<s>[INST] We'll have to look for work once we're back. [/INST]You're already thinking about that?</s>"
```

Looking at a sample like this makes it much more clear. The separators are `instruction_part = "[INST]"` and `response_part = "[/INST]"`

Now we can get training started. In our original script, we used `unsloth_save_pretrained_gguf` to save to gguf and quantize with Q4_K_M. You might be wondering why we are not using it this time. The reason is that it appears `unsloth_save_pretrained_gguf` will blow up my 24GB of VRAM, even though the training was successful. However, I found out that if you simply save as `safetensors`, then use llama.cpp to convert them to gguf then quantize manually, we could work around this VRAM limitation. Hence in our training script above, we are saving with `unsloth_save_pretrained_merged` instead, which merges LoRA and saves it into a directory `mistral_frieren`. Notice that if you ran unsloth before, it should have already created a `llama.cpp` directory under your current directory. If not, you can build [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) separately. Here is how to convert it to gguf then quantize it with Q4_K_M.

```bash
cd llama.cpp
python convert_hf_to_gguf.py ../mistral_frieren
cd build/bin
./llama-quantize ../../../mistral_frieren/Mistral-Small-24B-Instruct-bnb-4bit-2501-F16.gguf ../../../mistral_frieren/Mistral-Frieren.gguf Q4_K_M
```

Before we can add it to ollama, it's missing one more thing, which is its `Modelfile`. Thankfully it's pretty easy to borrow Mistral-Small's ModelFile.

```bash
cd ../../../mistral_frieren
ollama show --modelfile mistral-small:24b-instruct-2501-q4_K_M > Modelfile
vim Modefile # replace gguf file to the one we just created.
ollama create initialxy/mistral_frieren # add it to ollama
```

The rest works as usual. You can play with it in ollama or a frontend like [open-webui](https://github.com/open-webui/open-webui).

## Making Your Own BNB 4bit
In the above example, unsloth has already provided the bnb 4bit version of Mistral-Small. But what if they haven't? How can we create our own using a model of our choice? It's quite simple. Bnb 4bit is just [bitsandbytes' 4bit quantization](https://github.com/bitsandbytes-foundation/bitsandbytes). All you really have to do is load it using bnb 4bit and save it locally. Then later load from your local directory instead. I wrote a script based on how [unsloth loads in 4bit](https://github.com/unslothai/unsloth/blob/d1d15f1d14f1168837d29b9c08e9b6d63945d469/unsloth/models/loader.py#L331).

`tobnb4bit.py`
```python
#!/usr/bin/python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import argparse
import torch

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="Script to convert a model to bitsandbytes 4bits. " +
      "You still need to manully download its tokenizer.json etc."
   )
   parser.add_argument("model_id", help="Input model ID")
   parser.add_argument("output", help="Output JSONL file")
   args = parser.parse_args()

   major_version, _ = torch.cuda.get_device_capability()

   double_quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16 if not major_version >= 8 else torch.bfloat16,
   )

   model = AutoModelForCausalLM.from_pretrained(
      args.model_id,
      quantization_config=double_quant_config
   )

   model.save_pretrained(args.output)
```

As an example, I chose to use the [abliterated version of Mistral-Small instead](https://huggingface.co/huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated). Eg.

```bash
python tobnb4bit.py 'huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated' bnb4bit
```

This should create a new directory `bnb4bit` that contains bnb 4bit version of `Mistral-Small-24B-Instruct-2501-abliterated`. In the training script, load it locally. Eg.

```python
base_model_name = "./bnb4bit"
```

Unfortunately, I ran out of VRAM during the training step. I'm not sure why this abliterated version seems to take up just a bit too much VRAM. So I had to lower LoRA rank to `r=32` in order to get a successful training. The remaining steps are the same, just replace with new file names.