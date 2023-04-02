---
layout: post
author: initialxy
title: "Training Stable Diffusion LoRA with Kohya"
description: "Setting up LoRA Training with kohya_ss on AMD GPU."
category: "Lesson"
tags: [MachineLearning, StableDiffusion, LoRA]
---
{% include JB/setup %}

Since [my last post](/lesson/2023/02/01/training-stable-diffusion-concept-with-lora-on-amd-gpu), a lot has changed. So instead of adding updates to my previous post, I figured I could write a follow-up instead. A lot of quirks with [sd_dreambooth_extension](https://github.com/d8ahazard/sd_dreambooth_extension) that I mentioned last time have been fixed. It is now able to create standalone LoRA on its own without the hacks that I mentioned. However, I also want to give [kohya_ss](https://github.com/bmaltais/kohya_ss) another try and see if I can get it to work this time. Again, our main challenge here is to get it to work with an AMD GPU. Recall that last time I couldn't get it to work for a couple of issues: it had tons of hard-coded Windows path separators, which made it difficult to run on Linux, where PyTorch's ROCm build is available, and I couldn't get TensorFlow to work on AMD GPU. Things have certainly changed a lot in just a month or so. The good news is I managed to get it to work on Linux while running on AMD GPU. So I'd like to share my setup and some scripts that I wrote for myself.

<!--more-->

# Installation
Let's get [kohya_ss](https://github.com/bmaltais/kohya_ss) installed and running. This time around, it actually came with an Ubuntu install and launch script. So if you are on Ubuntu and Nvidia GPU, then you are ready to go! For me, I use Arch and AMD GPU, so I basically just followed its [Ubuntu script](https://github.com/bmaltais/kohya_ss/blob/master/ubuntu_setup.sh), except we don't install Python from `apt` and we need to install the [ROCm build of PyTorch](https://pytorch.org/get-started/locally/).

```bash
# Install git and Python 3 from your distro's repo first
cd git # go to a directory of your choice where you leave git repos
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
pip install --use-pep517 --upgrade -r requirements.txt

accelerate config
```

Now there are a couple of things we have to change at this point because of AMD GPU. First, AMD doesn't have xformer, so there's no point in installing that. The second is that instead of installing the default build of TensorFlow, we need to install [tensorflow-rocm](https://pypi.org/project/tensorflow-rocm/) instead, which is the ROCm build of TensorFlow. Fortunately, we don't actually need to use [AMD's Docker image](https://www.amd.com/system/files/documents/chapter5.1-tensorflow-rocm.pdf), which would have made it more complicated.

```bash
pip uninstall tensorflow # uninstall the default build of tensorflow first
pip install tensorflow-rocm
```

We are ready to launch, but before we can do that, we need to set an an environment variable `HSA_OVERRIDE_GFX_VERSION=10.3.0` like what we did for stable-diffusion-webui. Similarly, we can write a launch script that skips the default script's requirements check.

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0 # necessary to get pytorch and tensorflow to run with ROCm
cd ~/git/kohya_ss
source venv/bin/activate
python kohya_gui.py "$@"
```

Go to [http://localhost:7860/](http://localhost:7860/) and we got kohya_ss GUI. From here, it should be smooth sailing.

# LoRA Training
Following my previous post, I want to dive deeper into LoRA training. The first thing to notice is that kohya_ss sets up its training image differently than sd_dreambooth_extension. It needs `img`, `model`, and `log` directories for its inputs and outputs. It can help you create them under its **Tools** tab, but you can just do that on your own. Notably, the number of epochs to train, instance token, and class token are part of the image folder name. In my experiments, I found 5000 steps to be just about the right amount of training steps with the default 1e-5 **Learning rate** and cosine **LR scheduler**. This means you can compute the number of epochs by 5000 / number of images. eg. If I have 60 training images, I'd set my epochs to 83. So my image input folder name would be something like `83_shrug pose`, where "shrug" is the instance token and "pose" is the class token separated by a space. One very important thing to remember is that your instance token needs to be in all of the caption files. I wrote a script to help with this.

`prepend.py`
```python
#!/usr/bin/python
import argparse
import os

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Script to prepend a token to every .txt file under a directory."
  )
  parser.add_argument("dir", help="Directory where .txt files are found")
  parser.add_argument("token", help="Token to prepend. No , needed")
  args = parser.parse_args()

  files = os.listdir(args.dir)
  files = [f for f in files if os.path.isfile(os.path.join(args.dir, f)) and f.endswith(".txt")]
  for f in files:
    f = os.path.join(args.dir, f)
    with open(f, "r") as r:
      line = r.read()
    with open(f, "w") as w:
      w.write(args.token + ", " + line)
```

As for the rest of the settings, there are a few things we need to change in order for it to work for us.
* **Optimizer**: Lion. We don't have Adam for AMD.
* **Use xformers**: Uncheck. We don't have xformers for AMD.
* **Caption Extension**: Set this to `.txt` if your training images were prepred by stable-diffusion-webui. By default it will use `.caption`
* **Clip skip**: Set it to 2 for NAI based base model, 1 for everything else.
* **Shuffle caption**: Check. Supposedly makes outputs more evened out.

There are other settings you can fiddle with. I haven't spent too much time with LyCORIS. It is supposedly better, but I don't have an opinion on that yet. You will have to install [a1111-sd-webui-locon](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) in stable-diffusion-webui in order to use it. Now hit **Train model** and it should start training and output a `.safetensors` file under your `model` directory.

# More Scripts
Kohya_ss has a **Print training command** feature, where it prints out the command it uses to train in terminal. I love this. This means I can automate training without having to launch its GUI. I could chain a few trainings together before I go to sleep. It's great! I wrote a wrapper script to make things easier for me.

`train_lora.py`
```python
#!/usr/bin/python
import argparse
import math
import os
import re

KOHYA_DIR = "~/git/kohya_ss" # Enter your kohya_ss directory
EXT_RE = r".*\.png$"
DIR_RE = r"^(\d+)_.*"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Wrapper script to train LoRA using kohya_ss."
  )
  parser.add_argument(
    "--clip-skip",
    "-c",
    type=int,
    help="Training clip skip",
    default=1
  )
  parser.add_argument(
    "--output",
    "-o",
    help="Output file name",
    default="lora"
  )
  parser.add_argument('base', help="Base model")
  parser.add_argument('training', help="Training data directory")
  args = parser.parse_args()

  training_dir = os.path.abspath(args.training)
  img_dir = os.path.join(training_dir, "img")
  model_dir = os.path.join(training_dir, "model")
  log_dir = os.path.join(training_dir, "log")
  reg_dir = os.path.join(training_dir, "reg")

  training_img_dirs = [
    d
    for d in os.listdir(img_dir)
    if os.path.isdir(os.path.join(img_dir, d)) and re.match(DIR_RE, d)
  ]
  if not training_img_dirs:
    raise ValueError("Could not find directories under " + img_dir)

  steps = [
    int(re.match(DIR_RE, d)[1]) * len([f for f in os.listdir(os.path.join(img_dir, d)) if re.match(EXT_RE, f)])
    for d in training_img_dirs
  ]
  steps = sum(steps)

  training_cmd = f'accelerate launch --num_cpu_threads_per_process=2 "train_network.py" --pretrained_model_name_or_path="{os.path.abspath(args.base)}" --train_data_dir="{img_dir}" --resolution=512,512 --output_dir="{model_dir}" --logging_dir="{log_dir}" --network_alpha="1" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=5e-5 --unet_lr=0.0001 --network_dim=8 --output_name="{args.output}" --lr_scheduler_num_cycles="1" --learning_rate="0.0001" --lr_scheduler="cosine" --train_batch_size="1" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --cache_latents --optimizer_type="Lion" --bucket_reso_steps=64 --bucket_no_upscale --shuffle_caption --caption_extension=".txt" --lr_warmup_steps="{math.ceil(steps / 10)}" --train_batch_size="1" --max_train_steps="{steps}" --clip_skip={args.clip_skip}'
  training_cmd += f' --reg_data_dir="{reg_dir}"' if os.path.isdir(reg_dir) else ""

  os.system(
    "export HSA_OVERRIDE_GFX_VERSION=10.3.0 && " +
    f"cd {KOHYA_DIR} && " +
    "source venv/bin/activate && " +
    training_cmd
  )
```

It's a retty straightforward script. Use `--help` to see its instructions. But wait, I have some more scripts that I can share. Here is one that converts all image files to `.jpg` with [ImageMagick](https://wiki.archlinux.org/title/ImageMagick) in case you collected a bunch of images from the internet in various formats and want to normalize them.

`tojpg.py`
```python
#!/usr/bin/python
import argparse
import os
import re

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Script to convert all files under a directory to .jpg. Needs ImageMagick to be installed."
  )
  parser.add_argument("dir", help="Directory where image files are found")
  args = parser.parse_args()

  files = os.listdir(args.dir)
  files = [f for f in files if os.path.isfile(os.path.join(args.dir, f)) if not re.match(r".*\.(jpg|jpeg)$", f)]
  for f in files:
    f = f.replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")
    name = re.search(r"(.*)\.[a-zA-Z]+", f).group(1)
    os.system(f"convert -quality 95% {os.path.join(args.dir, f)} {os.path.join(args.dir, name)}.jpg")
```

Here is one that sequence file names in case you got a bunch of files from the internet with unreadable names and you just want them to look cleaner. It also helps when you are adding new images and want to make sure there's no collision in file names. (Use the `--start-index` option to start the sequence after the last sequenced file.)

`sequence.py`
```python
#!/usr/bin/python
import argparse
import os

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Script to sequence files under a directory."
  )
  parser.add_argument(
    "--start-index",
    "-s",
    type=int,
    help="Start index",
    default=0
  )
  parser.add_argument(
    "--prefix",
    "-p",
    help="Prefix of file names",
    default=""
  )
  parser.add_argument('dir', help="Directory under which files will be sequenced")
  args = parser.parse_args()

  files = os.listdir(args.dir)
  files = [f for f in files if os.path.isfile(os.path.join(args.dir, f))]
  files.sort()
  for i, f in enumerate(files):
    _, ext = os.path.splitext(f)
    f = f.replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")
    filename = f"{args.prefix}{args.start_index + i:03d}"
    os.system(f"mv {os.path.join(args.dir, f)} {os.path.join(args.dir, filename)}.{ext}")
```

Here is one that will deduplicate tokens (separated by `,`) and optionally filter a token in case you prepended tokens to all caption files but some of them may already have a token auto-generated, or you believe some auto-generated tokens are wrong.

`dedupe.py`
```python
#!/usr/bin/python
import argparse
import os

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Dedupe tokens in caption files. Optionally filter token."
  )
  parser.add_argument(
    "--filter",
    "-f",
    help="Token to remove",
    default=""
  )
  parser.add_argument("dir", help="Directory where .txt files are found")
  args = parser.parse_args()

  files = os.listdir(args.dir)
  files = [f for f in files if os.path.isfile(os.path.join(args.dir, f)) and f.endswith(".txt")]
  for f in files:
    f = os.path.join(args.dir, f)
    with open(f, "r") as r:
      line = r.read()
    with open(f, "w") as w:
      tokens = {t.strip() for t in line.split(",")}
      if args.filter:
        tokens = {t for t in tokens if t not in args.filter}
      w.write(", ".join(tokens))
```

Spelling and grammer of this post was checked by ChatGPT.