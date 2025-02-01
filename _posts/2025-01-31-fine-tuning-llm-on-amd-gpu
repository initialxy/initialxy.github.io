---
layout: post
author: initialxy
title: "Fine Tuning LLM on AMD GPU"
description: "Fine tuning Phi-4 on AMD consumer GPU with help from unsloth"
category: "Lesson"
tags: [MachineLearning, LLM, LoRA, Phi-4]
---
{% include JB/setup %}

I've been fascinated by open source LLM models and have been running them locally. I like to maintain full control of my ML models instead of relying on the cloud, simply because it's more fun that way. From my previous posts, you may be aware that I use AMD GPU on my Arch Linux (btw), so I'm gonna continue this trend of struggling to get training working on my AMD GPU. My most recent project is an attempt to recreate the character [Frieren from Frieren: Beyond Journey's End](https://en.wikipedia.org/wiki/Frieren). Why Frieren in particular? That's because it appears none of the open source LLM seem to be aware of this series at all. So any new behavior added can be attributed to what I do here instead of the base model's knowledge. Furthermore, Frieren has became a massively successful series internationally, which makes it somewhat easier to collect a lot of materials from the internet for training purpose. In order to accomplish this, I have two sub-goals:
* Train Frieren's style of speech with LoRA fine tuning.
* Add world knowledge to the fine tuned model using a RAG.

I've decided to break it down into two separate posts, since they cover different techniques and many many quirks. This post is focused on the first part, fine tuning for style.

<div class="preview_img_1" markdown="1">

![Frieren](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/frieren_cover.jpg)!

</div>

<!--more-->

## Objective
Alright, I will be honest. What I really want to accomplish is to create a bot that mimics myself so I can set it up to reply to my coworkers on Slack. Just like that episode in HBO's Silicon Valley.

[![Silicon Valley: Gilfoyle Made A Bot (Season 6 Episode 1 Clip) | HBO](https://img.youtube.com/vi/Y1gFSENorEY/0.jpg)](https://www.youtube.com/watch?v=Y1gFSENorEY)

To be fair, I'm sure I'm not the only one who's trying to do this. The recent trend of AI agents is going to that direction. But I want to do it completely locally on my home computer. But before I get there, I wanted to do a proof of concept project to see how feasible it is. So I decided to create Frieren, a character unknown to open source LLM models, along with her relevant knowledge. I've been following Matt Williams's YouTube channel, which has been enormously informative in my research.

[![19 Tips to Better AI Fine Tuning](https://img.youtube.com/vi/W2QuK9TwYXs/0.jpg)](https://www.youtube.com/watch?v=W2QuK9TwYXs)

It's generally understood that fine tune is a way to train for style and focus while RAG is the best way to bring in new knowledge into a LLM. In order to create Frieren, I will use both. So let's get started on fine tune first to get Frieren's style of speech.

## Hardware
Since my last post, I actually made some major upgrades to my computer hardware. Most of its core parts are actually refreshed.

| Component | Part name |
|-----------|-----------|
| CPU | AMD Ryzen 7 9800X3D |
| GPU | AMD Radeon RX 7900 XTX 24GB |
| RAM | 64GB (2 x 32GB) DDR5 6000Mhz |
| Storage | Samsung 980 Pro 1TB |

Here is a picture for good measure, since I'm also a build-a-pc enthusiast.

![Hardware](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/pc.jpg)!

I realized that AM4 CPU coolers are not always compatible with AM5. If your AM4 CPU cooler required a backplate then it won't work on AM5, because AM5 has a fixed backplate that you can't remove. On a different note, I swear I'm not actually an AMD fanboy. I was seriously considering picking up either a used RTX 3090 or Radeon RX 7900 XTX, both of which were about the same price. But RX 7900 XTX has somewhat better rasterized performance, so I figured it's better value (for gaming, not everything else) and AMD is generally more Linux friendly when it comes to drivers. So I went AMD again. In case you wonder what game I play. It's Minecraft Bedrock Edition. So it brings us here. Why do I insist on getting ML to work on AMD GPU? Why not just use [Google Colab](https://colab.research.google.com/) or [RunPod](https://www.runpod.io/) at least for training? Am I just a masochist? The answer is yes. Yes I am.

## Installation and Environment Setup
To get inference working is actually surprisingly easy this time. I just installed [ollama-rocm-git](https://aur.archlinux.org/packages/ollama-rocm-git) from AUR and it works. Great! For UI, I choose to use [open-webui](https://github.com/open-webui/open-webui), which was also pretty easy to install. I just installed it from `pip`. In case you need some commands for reference, here it is:

```bash
yay -S ollama-rocm-git # assuming you have yay on your Arch Linux
mkdir open-webui
cd open-webui
python3.11 -m venv venv # It's very important that you *don't* use Python 3.13, which has a backwards incompatible issue with C binary interop. Most of the libraries won't work with Python 3.13 as of writing.
source venv/bin/activate
pip install open-webui
open-webui serve
```

On a different terminal, run

```bash
ollama pull <pick a model from https://ollama.com/library>
ollama serve
```

Now open [http://localhost:8080](http://localhost:8080) and you should have inference ready to go.

The tricky part is to get things setup for training. AMD actually has [ROCm variants of most of the necessary libraries](https://github.com/ROCm), though at varying quality. I was following along AMD's own guide that they published recently: [Fine-tuning and inference using a single accelerator](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html), which was enormously helpful to get things started. However there were two open questions that I had:
* How should my training data look like for a specific model?
* How to run the output model in ollama?

That's when I discovered that [unsloth](https://github.com/unslothai/unsloth) can help with both. It can help normalize training data format for a specific model template and help you merge LoRA adapter into a gguf file that ollama can then pick up. So I decided to give unsloth a try. Unfortunately it looks like unsloth does not have [AMD GPU support](https://github.com/unslothai/unsloth/issues/37). However according to github user [sayanmndl21](https://github.com/unslothai/unsloth/issues/37#issuecomment-2445535450) most of the dependencies already have ROCm variants and they were able to get it to work with a small patch in code. So I decided to give it a try. Turns out it doesn't work for me, because the ROCm variant of xformers [only supports AMD's workstation GPUs](https://github.com/ROCm/composable_kernel/issues/1171#issuecomment-2305358524) and not consumer GPUs like my RX 7900 XTX. Hopefully this will change in the future, because as far as I can see, xformers is the only dependency blocking unsloth to work. If I were to remove xformers from unsloth's code, training will start and complete, but produce incorrect outputs. I made an attempted to replace it with SDP, but it was not a simple change and stretches beyond my domain knowledge. I believe [Axolotl](https://axolotl.ai/) does support AMD GPU, but looking at [Matt Williams' video](https://www.youtube.com/watch?v=lj44Bt9UxYQ), it appears quite quirky to get it to work. Perhaps I will try it in the future. But for now I'm following [AMD's guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html) and just borrow two helper functions from unsloth, training dataset normalization and gguf output. Here is now I got dependencies setup.

```bash
cd git # or wherever you keep your git repositories
git clone https://github.com/unslothai/unsloth.git
git clone --recurse https://github.com/ROCm/bitsandbytes.git
cd unsloth
python3.11 -m venv venv # Again, don't use Python 3.13
source venv/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.2.4 # As of writing I have ROCm 6.2 on my system
cd ../bitsandbytes
git checkout rocm_enabled_multi_backend
pip install -r requirements-dev.txt
cmake -DBNB_ROCM_ARCH="gfx1100" -DCOMPUTE_BACKEND=hip -S . # Find your GPU architecture from tools like amdgpu_top or rocminfo. Mine is gfx1100 = RX 7900 XTX
make # lol AMD's guide forgot this step, which means none of the binaries are actually built, resulting in some confusing errors later
python setup.py install
cd ../unsloth
pip install unsloth-zoo
```
That's it. Now if you want to train in a different directory just put `export PYTHONPATH=~/git/unsloth/` in your environment.

## Training Dataset Collection
We want to train for Frieren's style of speech, so what we need to collect is Frieren's dialogs. More specifically in Q&A format, where one character speaks to Frieren and she replies directly to it. I found [anime transcript of Frieren](https://transcripts.foreverdreaming.org/viewforum.php?f=2402) with a bit of Googling, and watched the first 7 episodes of Frieren while extracting her dialogs into a plain text file, which I named `frieren_dialog.txt`. This turned out to be an extremely time consuming and laborious process. I managed to collect 244 dialogs before giving up at the end of episode 7. Here is a snippet of it. Due to copyright concerns, I'm not going to share the entire raw file.

```
We'll have to look for work once we're back.
You're already thinking about that?
We've defeated the Demon King, but it's not over. We have the whole rest of our lives ahead of us.
I suppose you're right.
Frieren, the life ahead of you will surely be much longer than we can imagine.
Perhaps.
The king's going to erect statues of us in the plaza. I'm not sure they'll be able to faithfully recreate my handsome looks, though.
...
```
This text file is pretty simple, you can consider every odd line to be a user prompt and every even line to be a responses to the previous line. Next, we need to convert it to ShareGPT format so that unsloth can then normalize for a selected model template. This needs to be a JSONL file, which just means each line is one JSON object. Apparently the reason why they do this instead of just having a JSON array is so that a very large data file can be easily parsed, sampled, and split without having to load the whole file into memory. I guess that's fair. I wrote a Python script to do this. 

`tosharegpt.py`
```python
#!/usr/bin/python
import argparse
import json

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Script to alternating text dialog to ShareGPT JSONL format"
  )
  parser.add_argument("input", help="Input text file")
  parser.add_argument("output", help="Output JSONL file")
  args = parser.parse_args()

  with open(args.input) as f:
    lines = [l.strip() for l in f]

  conversations = []
  cur = []
  for i, l in enumerate(lines):
    if i % 2 == 0:
      cur = [{"from": "human", "value": l}]
    else:
      cur.append({"from": "gpt", "value": l})
      conversations.append(cur)

  with open(args.output, "w") as f:
    for l in conversations:
      j = json.dumps({"conversations": l})
      f.write(j)
      f.write("\n")
```

It's super straightforward to use. Run `python tosharegpt.py frieren_dialog.txt data.jsonl` and now you got a JSONL file that should look like this:

```
{"conversations": [{"from": "human", "value": "We'll have to look for work once we're back."}, {"from": "gpt", "value": "You're already thinking about that?"}]}
{"conversations": [{"from": "human", "value": "We've defeated the Demon King, but it's not over. We have the whole rest of our lives ahead of us."}, {"from": "gpt", "value": "I suppose you're right."}]}
{"conversations": [{"from": "human", "value": "Frieren, the life ahead of you will surely be much longer than we can imagine."}, {"from": "gpt", "value": "Perhaps."}]}
{"conversations": [{"from": "human", "value": "The king's going to erect statues of us in the plaza. I'm not sure they'll be able to faithfully recreate my handsome looks, though."}, {"from": "gpt", "value": "How self-serving of him. He only gave us ten copper coins when we left on our adventure."}]}
...
```

## LoRA Fine Tuning
Let's get training started. I picked Microsoft's Phi-4 as base model. The reason is simply that it's small enough to fit in my VRAM during training but also large enough to produce pretty reasonable responses most of the time. The great thing about using unsloth is that it will help normalize the data template for you, so if you were to pick a different base mode, you can do it quite easily. To get my training script working, I referenced [AMD's guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html) as well as [unsloth's Phi-4 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb).

`train.py`
```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from unsloth.save import unsloth_save_pretrained_gguf

base_model_name = "unsloth/phi-4-bnb-4bit"

# Load base model to GPU memory.
device = "cuda:0"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code = True).to(device)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return { "text" : texts, }

dataset = load_dataset("json", data_files="data.jsonl", split="train")

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

peft_config = LoraConfig(
    lora_alpha = 16,
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
        num_train_epochs = 4,
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
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

trainer_stats = trainer.train()

unsloth_save_pretrained_gguf(trainer.model, "ggufmodel", tokenizer, quantization_method="q4_k_m")

```

We start by loading unsloth's version of 4bit phi-4. You can find unsloth's repo mapping [here](https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/models/mapper.py#L527). When I loaded the original `microsoft/phi-4`, it just blew up VRAM. So I'm gonna stick with the 4bit version. Next we use unsloth's `get_chat_template` and `standardize_sharegpt` to normalize our training data for Phi-4. Then we configure `SFTTrainer` for LoRA fine tuning. Here are some of the notable parameters to watch out for:
* `r` AKA LoRA rank. Choose 16, 32 or 64. This is the number of layers LoRA will influence. Higher number will get you better results but needs more VRAM to train.
* `per_device_train_batch_size` is how big of a batch to use during training. Bigger batch size will use more VRAM, but enables faster training. Here, I set it to 1 due to an issue with `train_on_responses_only`, which I will elaborate a bit later.
* `num_train_epochs` and `max_steps` sets now many training steps will run. `num_train_epochs` effectively means how many times it will loop through your training dataset while `max_steps` simply sets the number of steps. You can use either parameter to set your training steps.
* `learning_rate` sets how much each step will change its parameters. In short, large learning rate value will converge faster but lower quality results, and smaller learning rate will take longer but produce finer quality results.

My attempt to explain the above parameters are extremely layman, please look up [full documentation](https://huggingface.co/docs/trl/en/sft_trainer) to learn more. My settings above actually deliberately under fits this model, which means it does not mimic Frieren as closely as it could. There's reason for this, which I will discuss in my next post regarding RAG.

Before we start training, I choose to use unsloth's `train_on_responses_only` to only fine tune on responses instead of both prompt and response. I feel this is important, because our goal is to *onl* mimic Frieren's style of speech and not the other characters'. However this function seems to cause a [bug](https://github.com/unslothai/unsloth/issues/1017) with `per_device_train_batch_size`, where trainer will produce this error.

```
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length.
```

What this means is that trainier things entries in dataset are not the same size, hence it is unable to batch them. It is asking you to either add padding or truncate them. Which is strange, because I thought we already enabled padding in tokenizer. Taking a look at its [source code](https://github.com/unslothai/unsloth-zoo/blob/1101ee09f9464e259163a12a4ed5735c06873769/unsloth_zoo/dataset_utils.py#L174), it appears to be doing some pretty hacky stuff that overwrites all instructions to -100. Given that, I'm not sure if there's a easy way to patch this, so I will just set `per_device_train_batch_size` to `1` in order to work around this issue.

Finally we can get training started with `trainer_stats = trainer.train()` and we borrow unsloth's `unsloth_save_pretrained_gguf` to save the results in gguf format with `Q4_K_M` quantization. Run this script with unsloth's `venv` activated.

```bash
python train.py
```

On my machine, this takes 244 steps and about 10 minutes to finish while consuming 16.6GB of VRAM, which is not bad at all.

![Training stats](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/training.jpg)!

Once it finishes, you should fine `ggufmodel` directory being created. Update `ggufmodel/Modelfile` to use `unsloth.Q4_K_M.gguf` instead. Here is my `Modelfile` for instance

```
FROM /home/initialxy/tmp/ggufmodel/unsloth.Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|im_start|><|system|><|im_sep|>{{ .System }}<|im_end|>{{ end }}{{ if .Prompt }}<|im_start|><|user|><|im_sep|>{{ .Prompt }}<|im_end|>{{ end }}<|im_start|><|assistant|><|im_sep|>{{ .Response }}<|im_end|>"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_sep|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
```

Now we import this model into ollama. Start ollama with `ollama serve` in another terminal then run.

```bash
cd ggufmodel
ollama create initialxy/frieren # or name it whatever you want
```

Start open-webui normally and we can see a new model `initialxy/frieren:latest` is available. Let's do a bit of a comparison.

![First test](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/model_test_1.jpg)!

Not bad. Sounds kind of like Frieren. Again, note that I deliverably under fit this model. Try to change your training steps with `num_train_epochs` and `max_steps` to get to a point where you think is right for you. It just takes some trial and error. I then created a character with some custom parameters and system prompts in open-webui.

![Frieren character](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/frieren_character.jpg)!

I chose to use 0.6 as temperture and increased its context length to 80k. Keep in mind that open-webui almost always defaults context length to 2k, and Phi-4 is supposed to use 16k context length. Now let's test it again.

![Second test](/static/images/2025-01-31-fine-tuning-llm-on-amd-gpu/model_test_2.jpg)!

Not bad, I like it.

## Closing
Try to play with the above mentioned parameters and see where you suits you best. Also try to collect as much *high quality* training data as possible will also improve quality. In the next post, I will discuss how we can combine it wih a RAG to supplement this model with world knowledge from Frieren: Beyond Journey's End in open-webui. It's not as straightforward as I hoped and required some hacky money patches.