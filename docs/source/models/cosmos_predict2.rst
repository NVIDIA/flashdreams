.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Cosmos-Predict2.5
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2511.00062" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://huggingface.co/nvidia/Cosmos-Predict2.5-2B" target="_blank" rel="noopener noreferrer">Model page</a>
     <a class="model-link-button" href="https://github.com/nvidia-cosmos/cosmos-predict2.5" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Cosmos-Predict2.5 is the latest member of the Cosmos World Foundation Models
(WFMs) family. It is a flow-based model that unifies Text2World, Image2World,
and Video2World into a single network and uses Cosmos-Reason1 - a Physical AI
reasoning vision language model - as its text encoder. The model is shipped in
2B and 14B sizes and post-trained for robotics and autonomous-vehicle tasks
through a curated 200M-clip pre-training corpus, model merging, and a new RL
algorithm.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://images.nvidia.com/aem-dam/Solutions/cosmos/cosmos-predict.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>
   <p class="model-footnote">
     Teaser video source:
     <a href="https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/">Cosmos-Predict2.5 project page</a>.
   </p>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/cosmos_predict2

Running the method
------------------

To run Cosmos-Predict2.5, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/cosmos_predict2 \
       flashdreams-run \
       cosmos2-t2v-2b-720p \
       --prompt "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."

For multi-GPU inference, use ``torchrun`` on top of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/cosmos_predict2 \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       cosmos2-t2v-2b-720p \
       --prompt "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."

For I2V, run with the following command:

.. code-block:: bash

   uv run --project integrations/cosmos_predict2 \
       flashdreams-run \
       cosmos2-i2v-2b-720p \
       --prompt "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation." \
       --image-path https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg \

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``cosmos2-t2v-2b-720p``
     - Cosmos-Predict2.5 2B T2V at 720p, prompt-only.
   * - ``cosmos2-i2v-2b-720p``
     - Cosmos-Predict2.5 2B I2V at 720p, prompt plus first-frame image.

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/cosmos_predict2 \
       flashdreams-run \
       cosmos2-t2v-2b-720p \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid zoomable">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/cosmos_predict2/cosmos2-t2v-2b-720p.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/cosmos_predict2/cosmos2-i2v-2b-720p.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         prompt: "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
         <br/>
         image: https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg
       </div>
     </div>
   </div>

Citation
--------

If you use Cosmos-Predict2.5, please cite the original work:

.. code-block:: bibtex

   @article{nvidia2025worldsimulationvideofoundation,
     title={World Simulation with Video Foundation Models for Physical AI},
     author={NVIDIA and Ali, Arslan and Bai, Junjie and Bala, Maciej and Balaji, Yogesh and Blakeman, Aaron and Cai, Tiffany and Cao, Jiaxin and Cao, Tianshi and Cha, Elizabeth and Chao, Yu-Wei and Chattopadhyay, Prithvijit and Chen, Mike and Chen, Yongxin and Chen, Yu and Cheng, Shuai and Cui, Yin and Diamond, Jenna and Ding, Yifan and Fan, Jiaojiao and Fan, Linxi and Feng, Liang and Ferroni, Francesco and Fidler, Sanja and Fu, Xiao and Gao, Ruiyuan and Ge, Yunhao and Gu, Jinwei and Gupta, Aryaman and Gururani, Siddharth and El Hanafi, Imad and Hassani, Ali and Hao, Zekun and Huffman, Jacob and Jang, Joel and Jannaty, Pooya and Kautz, Jan and Lam, Grace and Li, Xuan and Li, Zhaoshuo and Liao, Maosheng and Lin, Chen-Hsuan and Lin, Tsung-Yi and Lin, Yen-Chen and Ling, Huan and Liu, Ming-Yu and Liu, Xian and Lu, Yifan and Luo, Alice and Ma, Qianli and Mao, Hanzi and Mo, Kaichun and Nah, Seungjun and Narang, Yashraj and Panaskar, Abhijeet and Pavao, Lindsey and Pham, Trung and Ramezanali, Morteza and Reda, Fitsum and Reed, Scott and Ren, Xuanchi and Shao, Haonan and Shen, Yue and Shi, Stella and Song, Shuran and Stefaniak, Bartosz and Sun, Shangkun and Tang, Shitao and Tasmeen, Sameena and Tchapmi, Lyne and Tseng, Wei-Cheng and Varghese, Jibin and Wang, Andrew Z. and Wang, Hao and Wang, Haoxiang and Wang, Heng and Wang, Ting-Chun and Wei, Fangyin and Xu, Jiashu and Yang, Dinghao and Yang, Xiaodong and Ye, Haotian and Ye, Seonghyeon and Zeng, Xiaohui and Zhang, Jing and Zhang, Qinsheng and Zheng, Kaiwen and Zhu, Andrew and Zhu, Yuke},
     journal={arXiv preprint arXiv:2511.00062},
     year={2025}
   }
