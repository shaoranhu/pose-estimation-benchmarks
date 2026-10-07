# 模型调研与选型

检索日期：2026-09-14。成熟度以公开代码、官方权重、可复现评测和部署支持为依据，不只追逐榜单最高 AP。

## 单阶段多人模型

以下为官方 COCO keypoints val2017 指标，AP 使用 0–100 标度，不是本机测量。

| 模型 | 输入 | 参数量 M | FLOPs G | Pose AP | 用途判断 |
|---|---|---:|---:|---:|---|
| YOLO26n-pose | 640 | 2.9 | 7.6 | 57.2 | 内存和算力受限时的轻量基线 |
| YOLO26s-pose | 640 | 10.4 | 24.1 | 63.0 | 本轮初选，兼顾体积和精度 |
| YOLO26m-pose | 640 | 21.5 | 73.3 | 68.8 | 本机 GPU 上的精度对照 |
| YOLO26l-pose | 640 | 25.9 | 91.7 | 70.4 | 后续可评估，当前先控制实验规模 |
| YOLO26x-pose | 640 | 57.6 | 202.3 | 71.6 | 相对 l 增益有限，资源成本明显提高 |
| YOLO11s-pose | 640 | 9.9 | 23.2 | 58.9 | 老项目兼容基线，本轮不重复部署 |

来源：[YOLO26 官方姿态文档](https://docs.ultralytics.com/tasks/pose/)、[YOLO11 官方文档源文件](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo11.md)。YOLO26 表中规模为融合 Conv/BN、移除不用的训练分支后的推理模型；原始权重文件与加载前参数统计更大。不能直接用权重文件字节数除以 4 来反推此表参数量。

YOLO26 同时输出人体框和关键点，无需独立人体检测器。现有环境已支持该系列，因此工程落地成本低。选择它是本次环境和单阶段部署目标下的决定，并非证明它在所有姿态任务中最优。

RTMO 也是成熟的单阶段候选。官方 COCO-only 的 s/m/l AP 为 67.7/70.9/72.4；body7 多数据集训练的 l 为 74.8。官方仓库报告 V100、ONNXRuntime 延迟，不能与本机 PyTorch 延迟直接比较。本轮未核实其各档参数量，故不填猜测值。来源：[RTMO 官方模型表](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)。

## 两阶段模型及经典基线

| 模型 | 单人裁剪输入 | 规模口径 | 官方/论文 COCO val AP | 取舍 |
|---|---|---|---:|---|
| RTMPose-s | 256×192 | 轻量 CSPNeXt，额外需要人体检测器 | 71.6（COCO）；72.2（AIC+COCO） | 适合人数较少、需要更精细关键点的部署 |
| RTMPose-m | 256×192 | 论文姿态网络约 13.59M，检测器另计 | 74.6（COCO）；75.8（AIC+COCO） | 后续最值得增加的两阶段对照 |
| ViTPose-B | 256×192 | 论文表 9 报告 86M，检测器另计 | 75.8（单任务） | 精度优先，部署和算力预算更高 |
| ViTPose-L | 256×192 | 论文表 9 报告 307M，检测器另计 | 78.3（单任务） | 高精度研究基线，本轮不作为默认模型 |
| HRNet-W32 | 256×192 | ViTPose 论文对照表报告 29M，检测器另计 | 74.4 | 经典热图基线，适合与旧文献比较 |

来源：[MMPose RTMPose 模型表](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/README.md)、[RTMPose 论文](https://arxiv.org/pdf/2303.07399)、[ViTPose 官方仓库](https://github.com/ViTAE-Transformer/ViTPose)、[ViTPose 论文表 9](https://arxiv.org/pdf/2204.12484)。

两阶段方法先检测人体，再逐人裁剪估计姿态，成本随人数增加。RTMPose 论文的 75.8 AP 包含翻转测试；换成实时 RTMDet-nano 检测流程时论文给出 73.2 AP。ViTPose 仓库使用 person AP=56 的检测结果；论文的速度使用 A100、batch=64，属于吞吐量。上述 AP 可用于了解性能范围，但不是本轮无翻转、整图单阶段协议下的公平同机排名。

## 本轮决策规则

1. 先验证 n/s/m 在完整 COCO val2017 上的真实 AP、模型体积、显存、batch=1 延迟。
2. 默认倾向 s；如果 m 在本机仍有足够延迟余量且精度增益明确，将 m 作为本机精度优先推荐，s 保留为轻量部署方案。
3. 不用未经用户指定的 FPS 或 AP 阈值宣称“满足业务要求”；现场人数、遮挡、相机视角与目标设备仍需后续补充。
4. 本轮只验证预训练模型，不宣称实现了训练复现、3D 姿态或视频跟踪。

## 数据集与指标口径

使用 COCO val2017 全部 5,000 张图像及 `person_keypoints_val2017.json`。与仅取含关键点人体的 2,346 张 COCO-Pose 列表区分。保留无人体图像可以覆盖误检；COCOeval 使用官方 GT area、iscrowd、num_keypoints 和标准 OKS sigmas，不能与用框面积近似的内部 mAP 混为一谈。

来源：[COCO-Pose 官方说明](https://docs.ultralytics.com/datasets/pose/coco/)、[COCO 官方评测实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。本轮采用 COCOeval 默认 keypoints maxDets=20；预测保留最多 300 人，评测端按官方规则截断。预测分数使用人体框置信度，不做额外关键点分数融合或 OKS NMS。

代码及权重按上游许可证使用；YOLO26 官方分发标注 AGPL-3.0，商业集成前需核对适用授权。此处没有改变上游授权。
