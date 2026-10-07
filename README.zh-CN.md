> 发布副本：原始实验记录保留在此；公共入口默认使用独立 Python 环境和仓库内 `.cache`。完整预测文件与本机日志留在原实验目录，未包含在 Git 上传包中。

# 人体姿态估计：模型选型与 COCO 全量验证

完成日期：2026-09-14。任务范围：2D 多人、COCO 17 关键点，使用官方预训练权重进行验证；未训练或微调。

## 结论

**默认选择 YOLO26s-pose；本机精度优先时选择 YOLO26m-pose。** s 用约 10.36M 推理参数取得 62.28 AP；m 用约 21.54M 参数取得 68.20 AP，提升 5.92 个 AP 点，权重从 23.03 MiB 增至 46.77 MiB。n 为更小的部署备选，精度为 56.24 AP。由于尚未指定业务延迟与精度阈值，不能把默认选择视为全场景最优。

已完成 YOLO26n/s/m 各 5,000 张 COCO val2017 图像的验证（合计 15,000 次图像推理），保留 11,004 条原始人体标注及其官方忽略规则。各模型使用相同图像清单，含无人体图像；没有用训练集演示替代验证。

## 本机实测

| 模型 | 推理参数 M | 权重 MiB | Pose AP | AP50 | 平均延迟 ms | P95 ms | 峰值 allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n-pose | 2.93 | 7.51 | 56.24 | 81.33 | 8.85 | 16.19 | 92.2 |
| yolo26s-pose | 10.36 | 23.03 | 62.28 | 85.08 | 10.37 | 16.28 | 233.7 |
| yolo26m-pose | 21.54 | 46.77 | 68.20 | 88.71 | 17.87 | 18.84 | 195.4 |

AP/AP50 为 0–100 标度；原始 JSON 为 0–1。设备：RTX 4060 Laptop GPU（8GB），Windows 11，PyTorch CUDA 12.8，FP32，TF32 关闭，batch=1，640×640 letterbox。

延迟来自独立进程补测：每模型预热 50 次，固定前 100 张图像重复 3 轮，预先解码、CUDA 同步，包含 predict API、预处理、推理和后处理，不含磁盘读取、视频解码、绘图。它不是视频端到端 FPS。显存为 PyTorch 的 allocated 峰值，不含驱动/桌面占用，也不是 nvidia-smi 总显存；测量范围为该固定 100 图像测速集。

## 文件索引

- [模型调研](MODEL_SURVEY.md)：YOLO26/YOLO11、RTMO、RTMPose、ViTPose、HRNet 的证据与取舍。
- [实验记录](EXPERIMENT_LOG.md)：环境、下载问题、配置、结果和局限。
- [准确率汇总](results/full/summary.json)、[环境](results/full/environment.json)、[协议](results/full/protocol.json)、[图像 ID](results/full/image_ids.json)。
- `results/full/<模型>/`：完整 predictions.json、标准 cocoeval.txt、metrics.json、6 张骨架示例图。
- `results/isolated-latency/`：独立进程测速原始样本及统计。全量运行内的首次测速仍保留，但不作为此表的最终延迟。
- [完整运行日志](results/run.log)、[下载来源与哈希](results/download_manifest.json)。
- `results/smoke/`：8 张图像流程检查，仅检查代码连通性，不用于选型。

## 本机复现

在此 README 所在目录运行 PowerShell：

```powershell
# 已准备好的本机环境；不会修改原项目依赖
.\run-local.ps1 -Mode smoke
.\run-local.ps1 -Mode full
```

新结果写入 `results/smoke-rerun` 和 `results/full-rerun`，保留交付的原始实验。数据缓存位于本工作区 `.cache/`，包含 `val2017/`、`annotations/person_keypoints_val2017.json`、`weights/`。重新下载用 `.\run-local.ps1 -Mode prepare`，约 1.1GB 压缩数据加权重，需要网络访问。下载器按完整文件复用；失败的 `.part` 文件重新下载，不支持字节级续传。

发布副本入口使用独立 Python 环境；在新电脑上可按以下步骤准备：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/prepare_data.py --cache ./cache
python scripts/evaluate.py --cache ./cache --output ./results/reproduction
foreach ($model in @('yolo26m-pose','yolo26n-pose','yolo26s-pose')) {
    python scripts/benchmark_latency.py --cache ./cache --output ./results/reproduction-latency --model $model
}
```

参考 Python 3.12.14 和 NVIDIA CUDA 兼容驱动。此新环境安装流程为复现说明，未在第二台机器重跑；本机实测精确版本以 environment.json 为准。本报告和图表来自原始实验；发布副本保留这些已有文件。

## 评测细节与局限

- 使用 pycocotools.COCOeval 的 keypoints 模式，OKS 阈值 0.50:0.05:0.95、官方 area/iscrowd/num_keypoints 和 sigmas、maxDets=20。
- 预测 conf=0.001、max_det=300，人体框分数排序；不做额外 OKS NMS 或关键点置信度重打分。可视化只显示 conf≥0.25，和评测阈值不同。
- 单阶段整图输入、无翻转/多尺度测试；YOLO26 使用 end-to-end 分支。iou=0.7 被保存在配置中，但 end-to-end 分支不依赖传统框 NMS。
- 官方 n/s/m AP 为 57.2/63.0/68.8，本次为 56.24/62.28/68.20。图像范围、输入 padding、实现版本和评测流程存在差异；没有做隔离消融定位原因，因此不声称完全复现官方榜单。
- COCO 预训练模型在 COCO 验证集上的结果不能代表业务域泛化或未见过的测试集成绩。下一轮应在目标视角、遮挡与人数分布的数据上检查误检、漏检和关键点偏差。
- 完成的是预训练推理验证，不包括训练复现、TensorRT/ONNX 导出、移动端性能、3D、全身 133 点或视频跟踪。

官方来源：[YOLO26 姿态文档](https://docs.ultralytics.com/tasks/pose/)、[COCO-Pose 数据说明](https://docs.ultralytics.com/datasets/pose/coco/)。其他模型来源见调研文档。

## 可视化

![规模与准确率](results/tradeoff.png)

![YOLO26s 姿态示例](results/full/yolo26s-pose/example_000000000785.jpg)
