## Motion Inversion for Video Customization
<p align="center">
<br>
    <a href="https://arxiv.org/abs/2403.20193"><img src='https://img.shields.io/badge/arXiv-2403.20193-b31b1b.svg'></a>
    <a href='https://wileewang.github.io/MotionInversion/'><img src='https://img.shields.io/badge/Project_Page-MotionInversion-blue'></a>
    <a href='https://huggingface.co/spaces/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces(coming soon)-yellow'></a>
<!--     <a href='https://www.youtube.com/watch?v=Wq93zi8bE3U'><img src='https://img.shields.io/badge/Demo_Video-MotionDirector-red'></a> -->
<br>
</p>

[Luozhou Wang](https://wileewang.github.io/), [Guibao Shen](), [Yixun Liang](https://yixunliang.github.io/), [Xin Tao](http://www.xtao.website/), Pengfei Wan, Di Zhang, [Yijun Li](https://yijunmaverick.github.io/), [Yingcong Chen](https://www.yingcong.me)

HKUST(GZ), HKUST, Kuaishou Technology, Adobe Research.


 we present a novel approach to motion customization in video generation, addressing the widespread gap in the thorough exploration of motion representation within video generative models. Recognizing the unique challenges posed by video's spatiotemporal nature, our method introduces **Motion Embeddings**, a set of explicit, temporally coherent one-dimensional embeddings derived from a given video. These embeddings are designed to integrate seamlessly with the temporal transformer modules of video diffusion models, modulating self-attention computations across frames without compromising spatial integrity.  Furthermore, we identify the **Temporal Discrepancy** in video generative models, which refers to variations in how different motion modules process temporal relationships between frames. We leverage this understanding to optimize the integration of our motion embeddings.


<h4>Customize the motion in your videos with less than 0.5 million parameters and under 10 minutes of training time.</h4>

## 📰 News
* **[2024.04.03]** We released the configuration files, inference code sample.
* **[2024.04.01]** We will soon release the configuration files, inference code, and motion embedding weights. Please stay tuned for updates!
* **[2024.03.31]** We have released the project page, arXiv paper, and training code.

## 🚧 Todo List
* [x] Released code for the UNet3D model (ZeroScope, ModelScope).
* [x] Release detailed guidance for training and inference.
* [ ] Release Gradio demo.
* [ ] Release code for the Sora-like model (Open-Sora, Latte).



## Contents

* [Installation](#installation)
* [Training](#training)
* [Inference](#inference)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

<!-- * [Motion Embeddings Hub](#motion-embeddings-hub) -->

## Installation

```bash
# install torch
pip install torch torchvision

# install diffusers and transformers
pip install diffusers==0.26.3 transformers==4.27.4
```
Also, xformers is required in this repository. Please check [here](https://github.com/facebookresearch/xformers) for detailed installation guidance.

## Training

To start training, first download the [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) weights and specify the path in the config file. Then, run the following commands to begin training:

```bash
python train.py --config ./configs/train_config.yaml
```

Stay tuned for training other models and advanced usage!

## Inference
After cloning the repository, you can easily load motion embeddings for video generation as follows:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from models.unet.motion_embeddings import load_motion_embeddings

# load video generation model
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w",torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.enable_vae_slicing()

# load motion embedding
motion_embed = torch.load('path/to/motion_embed.pt')
load_motion_embeddings(pipe.unet, motion_embed)


video_frames = pipe(
    prompt="A knight in armor rides a Segway",
    num_inference_steps=30,
    guidance_scale=12,
    height=320,
    width=576,
    num_frames=24,
    generator=torch.Generator("cuda").manual_seed(42)
).frames[0]

video_path = export_to_video(video_frames)
video_path
```
Please note that it is recommended to use a noise initialization strategy for more stable outcomes. This strategy requires a source video as input. Click [here](./noise_init/) for more details.
Then you should pass the `init_latents` to `pipe` using the `latents` argument:
```python
video_frames = pipe(*,latents=init_latents).frames[0]
```

## Acknowledgement

* [MotionDirector](https://github.com/showlab/MotionDirector): We followed their implementation of loss design and techniques to reduce computation resources.
* [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w): The pretrained video checkpoint we used in our main paper.
* [AnimateDiff](https://github.com/guoyww/animatediff/): The pretrained video checkpoint we used in our main paper.
* [Latte](https://github.com/Vchitect/Latte): A video generation model with a similar architecture to Sora.
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): A video generation model with a similar architecture to Sora.

We are grateful for their exceptional work and generous contribution to the open-source community.

## Citation

 ```bibtex
@misc{wang2024motion,
      title={Motion Inversion for Video Customization}, 
      author={Luozhou Wang and Guibao Shen and Yixun Liang and Xin Tao and Pengfei Wan and Di Zhang and Yijun Li and Yingcong Chen},
      year={2024},
      eprint={2403.20193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` 

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date) -->
