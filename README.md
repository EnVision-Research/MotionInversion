<!-- <p align="center">
    <img src="./assets/readme/icon.png" width="250"/>
</p> -->
<!-- <div align="center">
    <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
    <a href="https://hpcaitech.github.io/Open-Sora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
    <a href="https://discord.gg/kZakZzrSUT"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
    <a href="https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-247ipg9fk-KRRYmUl~u2ll2637WRURVA"><img src="https://img.shields.io/badge/Slack-ColossalAI-blueviolet?logo=slack&amp"></a>
    <a href="https://twitter.com/yangyou1991/status/1769411544083996787?s=61&t=jT0Dsx2d-MS5vS9rNM5e5g"><img src="https://img.shields.io/badge/Twitter-Discuss-blue?logo=twitter&amp"></a>
    <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://hpc-ai.com/blog/open-sora-v1.0"><img src="https://img.shields.io/badge/Open_Sora-Blog-blue"></a>
</div> -->

## Motion Inversion for Video Customization

[Luozhou Wang](https://wileewang.github.io/), [Guibao Shen](), [Yixun Liang](https://yixunliang.github.io/), [Xin Tao](http://www.xtao.website/), Pengfei Wan, Di Zhang, [Yijun Li](https://yijunmaverick.github.io/), [Yingcong Chen](https://www.yingcong.me)

HKUST(GZ), HKUST, Kuaishou Technology, Adobe Research.


 we present a novel approach to motion customization in video generation, addressing the widespread gap in the thorough exploration of motion representation within video generative models. Recognizing the unique challenges posed by video's spatiotemporal nature, our method introduces **Motion Embeddings**, a set of explicit, temporally coherent one-dimensional embeddings derived from a given video. These embeddings are designed to integrate seamlessly with the temporal transformer modules of video diffusion models, modulating self-attention computations across frames without compromising spatial integrity.  Furthermore, we identify the **Temporal Discrepancy** in video generative models, which refers to variations in how different motion modules process temporal relationships between frames. We leverage this understanding to optimize the integration of our motion embeddings.


<h4>Customize the motion in your videos with less than 0.5 million parameters and under 10 minutes of training time.</h4>

## 📰 News

* **[2024.03.31]** We have released the project page, arXiv paper, and training code.
* **[2024.04.01]** We will soon release the configuration files, inference code, and motion embedding weights. Please stay tuned for updates!

## 🚧 Todo List
* [x] Released code for the UNet3D model (ZeroScope, ModelScope).
* [ ] Release detailed guidance for training and inference.
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
from diffusers import DDIMScheduler, DiffusionPipeline
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
Please note that it is recommended to use a noise initialization strategy for more stable outcomes. This strategy requires a source video as input. Here, we provide an example of its usage:
```python
import decord
from einops import rearrange
from noise_init import initialize_noise_with_blend
from utils.func_utils import tensor_to_vae_latent
from utils.ddim_utils import inverse_video
import torch

# Set decord's bridge to PyTorch and load video frames
decord.bridge.set_bridge('torch')
source_video = decord.VideoReader('path/to/source.mp4', width=576, height=320)[:]
source_video = (rearrange(source_video, "f h w c -> f c h w").unsqueeze(0) / 127.5 - 1).to(device, dtype=torch.float16)

# Convert to VAE latents and initialize noise for improved video generation
source_latents = tensor_to_vae_latent(source_video, pipe.vae)
init_latents = initialize_noise_with_blend(inverse_video(pipe, source_latents, 50), seed=0)
```
Then you should pass the `init_latents` to `pipe` using the `latents` argument:
```python
video_frames = pipe(*,latents=init_latents).frames[0]
```
We also offer a variety of noise initialization strategies for you to explore. Click [here](./noise_init/) to dig out more details.
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
