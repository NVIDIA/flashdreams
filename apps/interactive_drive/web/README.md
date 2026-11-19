# Web 驾驶座舱

FlashDreams 自带页面（`--mode webrtc` 启动后打印的地址）现在也有「开始录制／停止并保存／下载 ZIP」，直接使用后端同一端口，无需启动此目录的前端。默认保存到仓库 `artifacts/drive-exports`，可以用后端 `--export-dir` 指定位置；地图、目标轨迹和标定会随相机帧一起归档。

本目录座舱需要分别启动 FlashDreams 后端和 Vite 前端。`npm run dev` 只启动网页，不加载模型。

在仓库根目录启动后端：

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8089 -- \
  --web-ui --present-camera camera_front_wide_120fov \
  --scene /home/gjc/src/flashdreams/clipgt-residential-iter1500 \
  --views 7 --auxiliary-scale 0.5 --compile-denoise off \
  --denoise-steps 2 --chunk-frames 1 --direct-steering
```

另开终端启动前端：

```bash
cd apps/interactive_drive/web
npm run dev
```

打开 Vite 输出的地址（通常是 `http://localhost:5173`）。设置里的后端地址留空，会通过 Vite 将 `/api/webrtc/offer` 转发到 `127.0.0.1:8089`。修改默认代理目标可在启动时使用 `VITE_BACKEND_URL=http://后端地址:端口 npm run dev`。也可以在页面设置中填写明确的 HTTP 后端地址并点击重连，该地址会直接用于握手。

`--web-ui` 发送相机视频、真实车辆状态和原生 WorldScenario BEV 地图，无需生成旧的 ImGui 界面。`--no-ui` 仅提供视频，不包含座舱所需的遥测和地图。地图像素由模型适配器的 WorldScenario 渲染后传输，前端不绘制道路。

左侧七路预览来自模型最近生成块的真实画面，随每个生成块更新。列表可上下滚动，点击预览切换主画面；不提供全景拼接。

地图复用原版 BEV 正交相机、受控 ego 车辆池、航向向上与跟随位置，并完整显示方形地图，避免裁掉两侧道路。

顶部「开始录制 → 停止并归档」可下载原始相机帧、BEV、位姿、车辆状态、标定与控制时间轴的 ZIP。「视频制作」打开 `/comparison.html`，可先生成双侧 GT 样片，也可选择录制的模拟作为右侧。模板保留 local map，缺少前向长焦时使用带有 CROP 标识的正前 2× 中心裁切，并自动使用 `artifacts/drive-exports/assets/haval-h6/ego-model.npz` 中的本车模型。完整配置和离线命令见[导出与视频文档](../../../docs/source/demos/interactive_drive.md#web-recordings-and-comparison-videos)。

当前补齐的原始数据已解压到 `artifacts/jlu-123246900/jlu-20260714-123246900-raw-sixview`，对应地图场景为 `artifacts/jlu-123246900/scene-rgb-campus`，视频制作页默认使用这两个路径。异地运行时通过 `DRIVE_GT_PATH` / `DRIVE_SCENE_PATH` 指定绝对路径。若更改后端 `--export-dir`，启动前端时还需设置同路径的 `DRIVE_EXPORT_DIR`。导出接口仅允许本机访问；远程使用请 SSH 转发前端端口。

视频区域显示连接状态；右侧“后端已执行”显示最近一个生成块实际使用的油门、刹车和转向。车速来自后端，未收到遥测时显示 `—`。首次加载权重和生成画面需要等待；连接失败会显示原因。

- W：油门；S / 空格：刹车；A / D：左转 / 右转。
- 挡位通过 P/R/N/D 按钮或 `Shift + P/R/N/D` 选择；P/N/B 快捷键也可分别选择停车/空挡/倒挡。右侧显示后端确认的挡位。P 挡施加制动，N 挡切断油门，R/D 分别选择倒车/前进。
- R 或画面重置按钮：重新开始场景。
- 松键立即发送释放状态；切换窗口、页面隐藏和断开连接会释放输入。
- 手柄：左摇杆转向，右/左扳机油门/刹车，方向键上/下/左/右选择 D/R/P/N，Y 或 Start 重置。

控制在后端下一次生成时生效。已在生成中的视频块仍使用先前输入，因此模型推理耗时仍是操作到画面响应的下限。右侧回执可用于区分输入是否已生效和画面是否还在播放上一块。

降噪步数通过后端 `--denoise-steps` 设置。网页中的对应选项仅展示启动配置说明，不能在运行中切换。

```bash
npm test
npm run build
```

CPU 链路检查（仓库根目录，需已安装 pytest-asyncio、aiohttp、aiortc）：

```bash
uv run --no-sync --with pytest-asyncio pytest apps/interactive_drive/tests/test_web_cockpit.py -m ci_cpu -q
```

该检查通过真实 WebRTC 传输视频与控制，并驱动应用的车辆运动逻辑；图像使用 CPU 替身，不加载模型权重。

## 本地吉大六视角 RGB 场景

`jlu-jxu4trk0-20260714-123201.zip` 的兼容副本位于 `artifacts/jlu-six-camera/scene-rgb`。
原 ZIP 保留；副本沿用原始标定、197 帧位姿、地图、目标、首帧和逐相机描述，以 10 FPS 运行，不需要 `--lidar`。

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8089 --fps 10 -- \
  --web-ui --present-camera camera_front_wide_120fov \
  --scene /home/gjc/src/flashdreams/artifacts/jlu-six-camera/scene-rgb \
  --views 6 --fps 10 --auxiliary-scale 0.5 \
  --compile-denoise off --denoise-steps 4 --chunk-frames 3 \
  --world-model-seed 0 --direct-steering
```

两处 `--fps 10` 分别指定运行时视频帧率与应用场景时间步，保持传输、播放和模型轨迹使用同一时间轴。

前方广角保持 832×480，其余五路为 416×256，不补造第七路。模型视角 ID 按以下方向对应，实际相机内外参仍来自吉大数据：

| 原始通道 | 模型视角 ID | 模型相机名 |
| --- | ---: | --- |
| `CAM_FRONT` | 0 | `camera_front_wide_120fov` |
| `CAM_FRONT_RIGHT` | 1 | `camera_cross_right_120fov` |
| `CAM_BACK_RIGHT` | 2 | `camera_rear_right_70fov` |
| `CAM_BACK` | 3 | `camera_rear_tele_30fov` |
| `CAM_BACK_LEFT` | 4 | `camera_rear_left_70fov` |
| `CAM_FRONT_LEFT` | 5 | `camera_cross_left_120fov` |

这是实验性的方向映射，不代表真实视场角与模型名称中的角度相同，也不代表已经适配训练相机的外观先验。长时间生成画质需要另行验证。
已通过 61 项 CPU 检查，并在 RTX PRO 5000 72GB 上用本地权重完成两块、25 帧短生成；视频保存在 `artifacts/jlu-six-camera/smoke.mp4`。此短测不代表稳定运行速度或长时间画质。
转换脚本和检查产物保存在本地 `artifacts/jlu-six-camera/`；该目录被 Git 忽略，迁移到其他机器时需要一并复制场景。

2026-10-25 的复核修正了相机挂载外参到车体原点的转换、初始俯仰/横滚丢失、开场控制图及轨迹提前一帧，以及动力学积分依赖视频帧率的问题。积分现在使用不超过 1/120 秒的子步，视频帧率仍为 10 FPS；相关应用和集成 CPU 检查共 338 项通过。重启后端后生效，无需重新转换数据。

**画质验证仍未通过**：相同 seed 的两块生成中，辅助视角恢复全分辨率仍存在真实开场图之后的重绘；同一模型 latent 换回原版 Wan 解码器也没有消除它。源包说明其地图缺少首帧可见的斑马线和部分杆件几何，相机 ID 的对应也只是方向近似。这些数据/模型适配因素尚未完成隔离验证，不能将问题归因于六路数量，也不能将短生成成功当作官方 demo 等价画质。诊断与对照产物见本地 `artifacts/jlu-six-camera/diagnosis.md`。

另有 `artifacts/jlu-six-camera/scene-rgb-30fps` 对照场景：10 Hz 轨迹通过平移线性插值和旋转 Slerp 转为 30 FPS 控制，原采样逐一保留，没有增加实测数据。生成时需要同时将上述命令的场景路径和两处 FPS 改为对应值。该对照仍有开场重绘，不作为画质问题的解决方案。采集频率与 RGB 生成帧率可以不同，但仿真、控制和播放的时间轴必须一致；速度表的仿真秒也不等于未达到目标生成速度时的实际等待时间。

标定精度尚未得到独立验证。原包的前、右前、后视角叠加图已有部分道路投影不重合；可能涉及外参、地图/定位配准或高度，不能只靠参数文件匹配断定标定准确。原图按完整八项畸变参数转换到虚拟 F-theta 的检查能较好复现导出图，但这不是物理重投影精度证明。原始叠加图和检查结果位于 `artifacts/jlu-six-camera/calibration-check/`。

天气与风格也做了实测：此场景启用逐相机 `caption` 时，`sample.json` 顶层的通用 `prompt` 不参与生成。原始六路描述未统一天气和曝光；已另建 `scene-rgb-weather`（六路明确相同晴天、自然色彩和曝光）及 `scene-rgb-prose`（普通长描述入口）进行 49 帧同 seed 对照。两者都未保持开场图的明亮风格，前者反而更暗，因此没有替换上述默认场景。

绘制复核包括六路录制/实时路径同位姿对比、90 个已知三维点的独立投影检查，以及原包 WSM 视频直接控制生成。已知点到 CUDA 着色像素中心的最大距离小于 0.71 像素；原包控制视频生成仍有变暗和重绘，不能仅归因于当前绘制代码。侧视近处大框还存在曲线边缘/遮挡差异，两个渲染器并非逐像素等价。完整对照位于 `artifacts/jlu-six-camera/appearance-and-map-report.md`；可用 `uv run --no-sync python artifacts/jlu-six-camera/check_map_projection.py` 复跑仅使用地图渲染器的 GPU 检查。
