---
title: 'Cosmos3 Nano Sim Auto'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto">Checkpoint and model card</a>
  <a class="fd-button" href="https://research.nvidia.com/labs/cosmos-lab/cosmos3/">Cosmos 3 project</a>
  <a class="fd-button" href="https://blogs.nvidia.com/blog/open-world-models-physical-ai/">Blog</a>
</div>

Cosmos3 Nano Sim Auto generates driving video from an HD map, a prompt, and
opening images. [Interactive Drive](../demos/interactive_drive.md) supports one
or several cameras, scene switching, and optional generated LiDAR.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/cosmos3_nano_sim_auto/drive-7-view-lidar.avif" alt="Seven generated driving camera views and a LiDAR point cloud in a residential scene." />
</div>
<figcaption class="tiny-figcaption">
  FlashDreams Interactive Drive demo in-action, recorded with FlashDreams.
</figcaption>

## Quick Start

### Interactive Drive

Use Linux, Python 3.12, an NVIDIA GPU with **80 GB+ VRAM**, an R580+ driver,
**100 GB+ free storage**, a CUDA 13 toolkit (`nvcc`), and a C++ compiler.
The UI also needs Vulkan with CUDA interop on the server GPU.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[clone FlashDreams](../quickstart/index.md#install), then run from its root.
The checkout must contain `integrations_v2/cosmos3_nano_sim_auto`.

Your Hugging Face account needs access to the checkpoint above and
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes).

```bash
export HF_TOKEN="<your-hugging-face-token>"
uv sync --frozen --inexact --python 3.12 \
  --package flashdreams-cosmos3-nano-sim-auto --extra interactive-drive \
  --package flashdreams --group cuda13
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080
```

Open `http://<server-address>:8080/` (`localhost` for a local browser).

See [Interactive Drive controls](../demos/interactive_drive.md#controls) for
driving controls. Demo page also contains info on scene selection, rollout resets, and running without the UI (mandatory on GPUs without Vulkan).

#### Multi-GPU Interactive Drive

After installation above, launch one process per GPU on a single host
(four GPUs shown):

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 --no-python \
  flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 -- \
  --views 7 --parallel-decode
```

Set `--nproc_per_node` to the number of GPUs to use. Rank zero serves
`http://<server-address>:8080/`; camera decoding is distributed across GPUs.
Append `--lidar` after `--parallel-decode` to include generated LiDAR.
See [Interactive Drive controls](../demos/interactive_drive.md#controls) for how to drive around in the demo.

<div class="fd-cta-row">
  <a class="fd-button" href="#interactive-drive-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/interactive_drive.md#demo-arguments">Demo Arguments</a>
</div>

## Demo Presets

<a id="interactive-drive-presets"></a>

### Interactive Drive

| Preset | Description |
| --- | --- |
| `interactive-drive-cosmos3-nano-sim-auto` | One front-wide RGB view by default; prepared-scene driving with optional multiview and LiDAR. |

## Samples

<div class="model-media-grid zoomable">
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/cosmos3_nano_sim_auto/drive-7-view.avif" alt="Seven generated camera views of the same residential drive." />
    <figcaption class="tiny-figcaption">Seven RGB views.</figcaption>
  </div>
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/cosmos3_nano_sim_auto/drive-1-view-lidar.avif" alt="Generated front-camera video beside its LiDAR point cloud." />
    <figcaption class="tiny-figcaption">One RGB view and LiDAR.</figcaption>
  </div>
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/cosmos3_nano_sim_auto/drive-7-view-lidar.avif" alt="Seven generated driving cameras and a LiDAR point cloud." />
    <figcaption class="tiny-figcaption">Seven RGB views and LiDAR.</figcaption>
  </div>
</div>

## Developer Details

- [Integration source and setup details](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/cosmos3_nano_sim_auto)
- [LiDAR formats and timing](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/cosmos3_nano_sim_auto/LIDAR.md)
- [Cosmos 3 technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)

<hr>

## Citation

If you use Cosmos3 Nano Sim Auto, cite the original work:

NVIDIA et al. [**Cosmos 3: Omnimodal World Models for Physical AI**](https://arxiv.org/abs/2606.02800).
2026. arXiv:2606.02800. [DOI: 10.48550/arXiv.2606.02800](https://doi.org/10.48550/arXiv.2606.02800).

```bibtex
@misc{nvidia2026cosmos3omnimodalworld,
      title={Cosmos 3: Omnimodal World Models for Physical AI},
      author={{NVIDIA} and Aditi and Niket Agarwal and Arslan Ali and Jon Allen and Martin Antolini and Adeline Aubame and Alisson Azzolini and Junjie Bai and Maciej Bala and Yogesh Balaji and Josh Bapst and Aarti Basant and Mukesh Beladiya and Mohammad Qazim Bhat and Zaid Pervaiz Bhat and Dan Blick and Vanni Brighella and Han Cai and Tiffany Cai and Eric Cameracci and Jiaxin Cao and Yulong Cao and Mark Carlson and Carlos Casanova and Ting-Yun Chang and Yan Chang and Yu-Wei Chao and Prithvijit Chattopadhyay and Roshan Chaudhari and Chieh-Yun Chen and Junyu Chen and Ke Chen and Qizhi Chen and Wenkai Chen and Xiaotong Chen and Yu Chen and An-Chieh Cheng and Click Cheng and Xiu Chia and Jeana Choi and Chaeyeon Chung and Wenyan Cong and Yin Cui and Magdalena Dadela and Nalin Dadhich and Wenliang Dai and Joyjit Daw and Alperen Degirmenci and Rodrigo Vieira Del Monte and Robert Denomme and Sameer Dharur and Marco Di Lucca and Ke Ding and Wenhao Ding and Yifan Ding and Yuzhu Dong and Nicole Drumheller and Yilun Du and Aigul Dzhumamuratova and Aleksandr Efitorov and Hamid Eghbalzadeh and Naomi Eigbe and Imad El Hanafi and Hassan Eslami and Benedikt Falk and Jiaojiao Fan and Jim Fan and Amol Fasale and Sergiy Fefilatyev and Liang Feng and Francesco Ferroni and Sanja Fidler and Xiao Fu and Vikram Fugro and Prashant Gaikwad and TJ Galda and Katelyn Gao and Yihuai Gao and Wenhang Ge and Sreyan Ghosh and Arushi Goel and Vivek Goel and Akash Gokul and Rama Govindaraju and Jinwei Gu and Miguel Guerrero and Elfie Guo and Aryaman Gupta and Siddharth Gururani and Hugo Hadfield and Song Han and Ankur Handa and Zekun Hao and Mohammad Harrim and Ali Hassani and Nathan Hayes-Roth and Yufan He and Chris Helvig and Cyrus Hogg and Madison Huang and Michael Huang and Sophia Huang and Yufan Huang and Jacob Huffman and DeLesley Hutchins and Suneel Indupuru and Boris Ivanovic and Arihant Jain and Joel Jang and Ryan Ji and Yanan Jian and Dongfu Jiang and Jingyi Jin and Atharva Joshi and Nikhilesh Joshi and Pranjali Joshi and Andy Ju and Jaehun Jung and Weiwei Kang and Scott Kassekert and Jan Kautz and Ashna Khetan and Julia Kiczka and Slawek Kierat and Gwanghyun Kim and Kuno Kim and Sunny Kim and Kezhi Kong and Xin Kong and Zhifeng Kong and Tomasz Kornuta and Egor Krivov and Hui Kuang and Saurav Kumar and Chia-Wen Kuo and George Kurian and Wojciech Kutak and JF Lafleche and Himangshu Lahkar and Omar Laymoun and Jayjun Lee and Sanggil Lee and Gabriele Leone and Boyi Li and Freya Li and Jiajun Li and Jinfeng Li and Ling Li and Pengcheng Li and Shangru Li and Tingle Li and Xiaolong Li and Xuan Li and Zhaoshuo Li and Zhiqi Li and Hao Liang and Maosheng Liao and Chen-Hsuan Lin and Tsung-Yi Lin and Ming-Yu Liu and Sifei Liu and Zihan Liu and Hai Loc Lu and Xiangyu Lu and Alice Luo and Ruipu Luo and Wenjie Luo and Jiangran Lyu and Martin Ding Ma and Nic Ma and Qianli Ma and Dawid Majchrowski and Louis Marcoux and Miguel Martin and Qing Miao and Ashkan Mirzaei and Shreyas Misra and Kaichun Mo and Durra Mohsin and Hyejin Moon and Pawel Morkisz and Saeid Motiian and Kirill Motkov and Seungjun Nah and Yashraj Narang and Deepak Narayanan and Thabang Ngazimbi and Julian Ouyang and Shubham Pachori and David Page and Yatian Pang and Sehwi Park and Mahesh Patekar and Mostofa Patwary and Marco Pavone and Trung Pham and Wei Ping and Soha Pouya and Shrimai Prabhumoye and Varun Praveen and Delin Qu and Hesam Rabeti and Morteza Ramezanali and Marilyn Reeb and Xuanchi Ren and Kristen Rumley and Wojciech Rymer and Jun Saito and Yeongho Seol and John Shao and Piyush Shekdar and Tianwei Shen and Humphrey Shi and Min Shi and Stella Shi and Kevin Shih and Mohammad Shoeybi and Mateusz Sieniawski and Shuran Song and Alexander Sotelo and Amir Sotoodeh and Sunil Srinivasa and Vignesh Srinivasakumar and Bartosz Stefaniak and Rahul Heinrich Steiger and Shangkun Sun and Jiaxiang Tang and Shitao Tang and Yangyang Tang and Yue Tang and Tolou Tavakkoli and Kayley Ting and Krzysztof Tomala and Wei-Cheng Tseng and Jibin Varghese and Sergei Vasilev and Thomas Volk and Raju Wagwani and Roger Waleffe and Andrew Z. Wang and Boxiang Wang and Haoxiang Wang and Qiao Wang and Shihao Wang and Shijie Wang and Ting-Chun Wang and Yan Wang and Yu Wang and Rohit Watve and David Wehr and Fangyin Wei and Xinshuo Weng and Jay Zhangjie Wu and Kedi Wu and Hongchi Xia and Summer Xiao and Tianjun Xiao and Kevin Xie and Daguang Xu and Jiashu Xu and Mengyao Xu and Ruqing Xu and Xingqian Xu and Yao Xu and Dinghao Yang and Dong Yang and Hans Yang and Xiaodong Yang and Xuning Yang and Yichu Yang and Yurong You and Zhiding Yu and Hao Yuan and Simon Yuen and Xiaohui Zeng and Pengcuo Zeren and Cindy Zha and Haotian Zhang and Jenny Zhang and Jing Zhang and Liangkai Zhang and Paris Zhang and Shun Zhang and Xuanmeng Zhang and Zhizheng Zhang and Ann Zhao and Yilin Zhao and Yuliya Zhautouskaya and Charles Zhou and Fengzhe Zhou and Shilin Zhu and Yuke Zhu and Dima Zhylko and Artur Zolkowski},
      year={2026},
      eprint={2606.02800},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={10.48550/arXiv.2606.02800},
      url={https://arxiv.org/abs/2606.02800},
}
```
