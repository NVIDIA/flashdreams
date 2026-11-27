---
title: 'FlashDreams ROS 2 启动与关闭'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashDreams ROS 2 启动与关闭

这份文档对应 Pro6000 上实测的 JLU 校园六相机配置。模型用 GPU 0，ROS 2 Humble 使用本机域 96。桌面端通过 VS Code Remote - SSH 转发 `8765`，在 Foxglove 看画面和车辆状态，或在礁标用手柄驾驶。

## 1. 准备 ROS 包

在 Pro6000 的 Bash 终端执行。命令使用仓库中已安装的 `.venv`、本地权重 `Cosmos3-Nano-Sim-Auto`、本地 JLU 场景和 `.cache/foxglove-ros`。

```bash
cd /media/kemove/JieBro4t/GJC/flashdreams
source /opt/ros/humble/setup.bash
colcon --log-base /tmp/flashdreams-ros-build/log build --base-paths ros2 \
  --build-base /tmp/flashdreams-ros-build/build \
  --install-base /tmp/flashdreams-ros-build/install
```

首次启动、ROS 包变更后，或系统清理了 `/tmp/flashdreams-ros-build` 时执行编译。

## 2. 启动模型和 ROS 网关

启动命令从仓库根目录运行。两个进程都作为当前用户的 systemd 服务在后台运行。

```bash
cd /media/kemove/JieBro4t/GJC/flashdreams
systemd-run --user --collect --unit=flashdreams-ros-model \
  --property=KillSignal=SIGINT --working-directory="$PWD" \
  --setenv=CUDA_VISIBLE_DEVICES=0 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  /bin/bash -c '
    source .venv/bin/activate &&
    install -d -m 700 "${XDG_RUNTIME_DIR:-/tmp}/flashdreams-${UID}" &&
    exec flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
      --mode null --presentation-mode on_demand --timeout unbound -- \
      --no-ui --ros-socket "${XDG_RUNTIME_DIR:-/tmp}/flashdreams-${UID}/drive.sock" \
      --scene artifacts/jlu-123246900/scene-rgb-campus \
      --checkpoint Cosmos3-Nano-Sim-Auto --views 6 --auxiliary-scale 0.5 \
      --fps 10 --chunk-frames 2 --compile-denoise off --denoise-steps 4 \
      --world-model-seed 0 --initial-speed-mps 0 --road-height-primitives 30 32
  '
```

```bash
systemd-run --user --collect --unit=flashdreams-remote-joy \
  --working-directory="$PWD" \
  --setenv=ROS_LOCALHOST_ONLY=1 --setenv=ROS_DOMAIN_ID=96 \
  /bin/bash -c '
    source /opt/ros/humble/setup.bash &&
    source .cache/foxglove-ros/setup.bash &&
    source /tmp/flashdreams-ros-build/install/local_setup.bash &&
    exec ros2 launch flashdreams_bridge remote_joy.launch.py address:=127.0.0.1 port:=8765
  '
```

模型和 ROS 桥都从用户运行目录定位 `drive.sock`。日志里看到 `Connected to model` 后，发来第一条有效手柄消息即初始化回合。首帧发布时桥会记录 `Initial observation`，之后每个稳态生成块包含 8 帧。

## 3. 转发端口并连接

1. 在操作电脑用 VS Code Remote - SSH 连到 Pro6000，在「端口」面板转发远程 `8765`。
2. 在 Foxglove 选择 **Foxglove WebSocket**，连接 `ws://127.0.0.1:8765`。如果 VS Code 分配了不同的本地端口，以「端口」面板的本地端口为准。礁标若提示「无效的本地回放地址」，先用 Foxglove 客户端验证连接。礁标的连接和回放说明见[远程操控配置](interactive_drive_ros2_design.md#从礁标通过-websocket-遥控)。
3. 导入仓库里的 `ros2/examples/jlu_wireframe_layout.json`。3D 面板显示地图和车辆线框。上排中央可切换「前视 RGB」与「前视 WSM」，首次推进后可看到模型实际使用的 WSM 条件图。六路远程预览使用各自的 `image_raw/compressed` 和 `camera_info`。
4. 用礁标驾驶时，在「远程操纵」打开手柄模式，发布 `/joy`，类型为 `sensor_msgs/msg/Joy`。RT 给油，LT 制动，左摇杆转向，十字键上挂 D、下挂 R，Y 重置回合。旧布局若报 `/joy` 类型冲突，把方向键模式的主题改回 `/cmd_vel`，保存后重新连接。手柄模式使用 `/joy`。

## 4. 查看日志和手柄消息

```bash
systemctl --user status flashdreams-ros-model.service flashdreams-remote-joy.service
journalctl --user -u flashdreams-ros-model.service -u flashdreams-remote-joy.service -n 100 -f
```

Foxglove 的 Log 面板可以查看 `/rosout`。服务器端可以在新终端确认手柄消息和仿真状态。

```bash
source /opt/ros/humble/setup.bash
source /tmp/flashdreams-ros-build/install/local_setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=96
ros2 topic echo /joy sensor_msgs/msg/Joy --once --qos-reliability best_effort
ros2 topic echo /diagnostics --once
```

## 5. 关闭

```bash
systemctl --user stop flashdreams-remote-joy.service flashdreams-ros-model.service
systemctl --user is-active flashdreams-remote-joy.service flashdreams-ros-model.service
ss -ltn '( sport = :8765 )'
```

服务停止后，`is-active` 显示两个 `inactive`，端口检查只剩表头。下一次从第 2 步启动新回合。
