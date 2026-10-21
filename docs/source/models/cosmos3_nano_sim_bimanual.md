---
title: 'Cosmos3 Nano Sim Bimanual'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual">Checkpoint and model card</a>
  <a class="fd-button" href="https://research.nvidia.com/labs/cosmos-lab/cosmos3/">Cosmos 3 project</a>
  <a class="fd-button" href="https://blogs.nvidia.com/blog/open-world-models-physical-ai/">Blog</a>
</div>

Cosmos3 Nano Sim Bimanual generates robot-manipulation video from an opening
image, a prompt, and controller actions. Try it with
[HandDreams](../demos/hand_dreams.md).

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/demo_clips/handdreams.avif" alt="Generated robot hands folding a T-shirt in HandDreams." />
</div>
<figcaption class="tiny-figcaption">
  HandDreams demo in-action, recorded with FlashDreams.
</figcaption>

## Quick Start

### HandDreams

Use Linux x86_64/aarch64 with glibc 2.35+, OpenSSL, Python 3.12 venv support,
and an NVIDIA GPU with CUDA, Vulkan, and **NVENC**. RTX PRO 6000 has been tested. A GB300-only host cannot run this CloudXR setup.
Allow about **61 GB** for model shards and the merged cache, plus dependencies,
decoder weights, and scenes. Use Chrome or Chromium with graphics acceleration.
No headset or physical controllers are required for the steps below.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[clone FlashDreams](../quickstart/index.md#install), then run from its root.
The checkout must contain `integrations_v2/cosmos3_nano_sim_bimanual`.
Your Hugging Face account needs access to the checkpoint above and
[agibot-world/AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta).

[Setup isaac-teleop](../demos/hand_dreams.md#setup-isaac-teleop) if not done already.

```bash
export HF_TOKEN="<your-hugging-face-token>"
# Note: `--accept-eula` implies that you have already accepted the 'Isaac Teleop/CloudXR eula' terms. If not, follow the setup instructions for isaac-teleop setup.
uv run --no-project --python 3.12 python \
  integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py server \
  --address 127.0.0.1 --accept-eula
```

For a remote browser, replace `127.0.0.1` with the server's reachable LAN IPv4.
First startup installs dependencies, downloads the model and scenes, and warms
up the model. Keep the terminal running.

Connect [virtual controllers](../demos/hand_dreams.md#connect-virtual-controllers)
or [physical controllers](../demos/hand_dreams.md#connect-physical-controllers),
select a scene, and start controlling the hands.

<div class="fd-cta-row">
  <a class="fd-button" href="#handdreams-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/hand_dreams.md#controls">Controls</a>
  <a class="fd-button" href="../demos/hand_dreams.md#demo-arguments">Demo Arguments</a>
</div>

## Demo Presets

<a id="handdreams-presets"></a>

### HandDreams

| Preset | Description |
| --- | --- |
| `handdreams` | GA checkpoint, four denoising steps, TAEHV2 Super decoder, 640 × 352 output. |

The launcher selects this preset and prepares dependencies. Default scenes are
**Fold the T-shirt**, **Fold the shorts**, and **Open the tissue box**.
No checkpoint override or S3 credentials are needed.

## Developer Details

- [Integration and checkpoint details](https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/cosmos3_nano_sim_bimanual)
- [Native-window and physical Quest controller setup](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md)
- [Cosmos 3 technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)

<hr>

## Citation

If you use Cosmos3 Nano Sim Bimanual, cite the original work:

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
