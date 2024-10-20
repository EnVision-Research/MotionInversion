## Motion Inversion for Video Customization
<h4>Customize the motion in your videos under 10 minutes of training time.</h4>
<br>
    <a href="https://arxiv.org/abs/2403.20193"><img src='https://img.shields.io/badge/arXiv-2403.20193-b31b1b.svg'></a>
    <a href='https://wileewang.github.io/MotionInversion/'><img src='https://img.shields.io/badge/Project_Page-MotionInversion-blue'></a>
    <a href='https://huggingface.co/spaces/'><img src='https://huggingface.co/spaces/ziyangmai/MotionInversion'></a>
<!--     <a href='https://www.youtube.com/watch?v=Wq93zi8bE3U'><img src='https://img.shields.io/badge/Demo_Video-MotionDirector-red'></a> -->
<br>

[Luozhou Wang*](https://wileewang.github.io/), [Ziyang Mai*](https://ziyang1106.github.io/), [Guibao Shen](), [Yixun Liang](https://yixunliang.github.io/), [Xin Tao](http://www.xtao.website/), Pengfei Wan, Di Zhang, [Yijun Li](https://yijunmaverick.github.io/), [Yingcong Chen†](https://www.yingcong.me)

HKUST(GZ), HKUST, Kuaishou Technology, Adobe Research.

*Indicates Equal Contribution. †Indicates Corresponding Author.

In this work, we present a novel approach for motion customization in video generation, addressing the widespread gap in the exploration of motion representation within video generative models. Recognizing the unique challenges posed by the spatiotemporal nature of video, our method introduces **Motion Embeddings**, a set of explicit, temporally coherent embeddings derived from a given video. These embeddings are designed to integrate seamlessly with the temporal transformer modules of video diffusion models, modulating self-attention computations across frames without compromising spatial integrity. Our approach provides a compact and efficient solution to motion representation, utilizing two types of embeddings: a **Motion Query-Key Embedding** to modulate the temporal attention map and a **Motion Value Embedding** to modulate the attention values. Additionally, we introduce an inference strategy that excludes spatial dimensions from the Motion Query-Key Embedding and applies a differential operation to the Motion Value Embedding, both designed to debias appearance and ensure the embeddings focus solely on motion. Our contributions include the introduction of a tailored motion embedding for customization tasks and a demonstration of the practical advantages and effectiveness of our method through extensive experiments.

<!-- insert a teaser gif -->
<img src="assets/mi.gif"  width="400" />



## 📰 News
* **[2024.10.20]** Hugging face demo is ready. Click [here](https://huggingface.co/spaces/ziyangmai/MotionInversion).
* **[2024.10.15]** We improve the structure of motion embedding and obtain better performance. Check our latest [paper](https://arxiv.org/abs/2403.20193).
* **[2024.04.03]** We released the configuration files, inference code sample.
* **[2024.04.01]** We will soon release the configuration files, inference code, and motion embedding weights. Please stay tuned for updates!
* **[2024.03.31]** We have released the project page, arXiv paper, and training code.

## 🚧 Todo List
* [x] Released code for the UNet3D model (ZeroScope, ModelScope, VideoCrafter2).
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
pip install -r requirements.txt
```
Also, xformers is required in this repository. Please check [here](https://github.com/facebookresearch/xformers) for detailed installation guidance.

## Training

To start training, first download the [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) weights and specify the path in the config file. Then, run the following commands to begin training:

```bash
python train.py --config ./configs/config.yaml
```
We provide a sample config file in [config.py](./configs/config.yaml). 
Note for various motion types and editing requirements, selecting the appropriate loss function impacts the outcome. In scenarios where only the camera motion from the source video is desired, without the need to retain information about the objects in the source, it is advisable to employ [DebiasedHybridLoss](./loss/debiased_hybrid_loss.py). Similarly, when editing objects that undergo significant deformation, [DebiasedTemporalLoss](./loss/debiased_temporal_loss.py) is recommended. For straightforward cross-categorical editing, as described in [DMT]('https://diffusion-motion-transfer.github.io/'), utilizing [BaseLoss](./loss/base_loss.py) function suffices.

## Inference
After training, you can easily load motion embeddings for video generation as follows:

```
python inference.py
```

If you want to test more checkpoints, don't forget to modify the paths asigned in the code, including ```embedding_dir``` and ```video_round``` .

## Acknowledgement

* [MotionDirector](https://github.com/showlab/MotionDirector): We followed their implementation of loss design and techniques to reduce computation resources.
* [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w): The pretrained video checkpoint we used in our main paper.
* [AnimateDiff](https://github.com/guoyww/animatediff/): The pretrained video checkpoint we used in our main paper.
* [Latte](https://github.com/Vchitect/Latte): A video generation model with a similar architecture to Sora.
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): A video generation model with a similar architecture to Sora.
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): A video generation model with a similar architect ure to ZeroScope and stronger ability for high-quality generation.
* [VideoCrafter2 (diffuser checkpoint)](https://hf-mirror.com/adamdad/videocrafterv2_diffusers): A checkpoint file that can seamlessly integrate in our framework with one line code.
* [Textual Inversion](https://github.com/rinongal/textual_inversion): A image generation method that provides simple yet efficient way for attention injection.

We are grateful for their exceptional work and generous contribution to the open-source community.

## Citation

 ```bibtex
@misc{wang2024motioninversionvideocustomization,
      title={Motion Inversion for Video Customization}, 
      author={Luozhou Wang and Ziyang Mai and Guibao Shen and Yixun Liang and Xin Tao and Pengfei Wan and Di Zhang and Yijun Li and Yingcong Chen},
      year={2024},
      eprint={2403.20193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.20193}, 
}
``` 

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date) -->
