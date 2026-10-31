# 研究方案 B（共用部分）：ABIDE I/II 静息态 fMRI 共用数据流程与四个课题的边界

**状态**：v0.1（2026-10-30）。阶段 0 进行中：数据下载（`scripts/B01_download_dcan.py`）已启动，质控阈值与特征定义在第 5、6 节预先写定。第 9 节可行性检查完成后冻结为 v1.0；冻结后的改动登记在第 12 节变更日志，并在论文中标注为敏感性或事后分析。
**与课题 A 的关系**：样本纳入标准、表型表（`data/interim/abide_pheno_harmonized_20261014.csv`）、FastSurfer 体积和常模框架（GAMLSS、站点重新居中）全部沿用课题 A，不重复建设。
**子课题**：B1 结构与功能常模偏离（`protocol_B1_常模偏离_多尺度解释.md`）；B2 μCAPs；B3 多重宇宙稳健性；B4 梯度与有向层级。B2–B4 的 protocol 分别在各自的启动门槛通过后再写（第 10 节）。

---

## 1. 为什么设共用流程

四个课题用同一数据库，审稿人对站点效应、头动、对照偏少这些问题的质疑是相同的。先用一条有据可查的流程把这些共性问题处理掉，四篇才能互相支撑，而不是在同一处一起失败。各课题都只能在本流程之上增加内容，不能各自改质控或预处理；需要偏离时，写进本文件第 12 节并说明理由。

## 2. 数据来源（2026-10-30 逐项核查 FCP-INDI S3，证据见 `results/abide_derivatives_inventory_20261030.json`）

| 数据 | 位置 | 覆盖 | 用途 |
|---|---|---|---|
| **DCAN（ABCD-HCP BIDS 管线）衍生数据** | `ABIDE/Derivatives/DCAN/`、`ABIDE2/Derivatives/DCAN/` | 有分区时间序列：ABIDE I 1,089 人、ABIDE II 991 人 | **主管线**。两个队列用同一条预处理，ABIDE I 发现、ABIDE II 复现 |
| ├ 分区时间序列 | `func/*atlas-<图谱>_desc-filtered_timeseries.ptseries.nii` | 4 个图谱：HCP2016（Glasser 360 + 19 个皮层下）、Gordon2014（333 + 19）、Power2011、Markov2012 | 主分析用 HCP2016；图谱敏感性用 Gordon2014 |
| ├ 剔帧掩膜 | `func/*desc-filtered_motion_mask.mat` | 同上 | FD 阈值 0–0.5 mm（步长 0.01）下每个阈值对应的剔帧向量、剩余帧数和剩余秒数 |
| ├ 皮层厚度 | `anat/*space-fsLR32k_thickness.dscalar.nii` | 同上 | B1 的结构偏离 |
| └ 稠密时间序列 | `func/*desc-filtered_timeseries.dtseries.nii`（单个约 0.35–0.7 GB） | 同上 | 仅 B2（种子内体素/顶点层面）需要，按需流式下载后只保留种子区，不整体落盘 |
| PCP 预处理 | `ABIDE_Initiative/Outputs/{cpac,dparsf,niak,ccs}/{filt,nofilt}_{global,noglobal}/rois_*` | **仅 ABIDE I**；CC200 在 16 种组合中均为 1,100–1,102 人 | B3 的多重宇宙（其余 6 个图谱的人数尚未统计） |
| 不使用 | `ABIDE2/Outputs/fmriprep`（仅 586 人，含 crash 日志，不完整）；`ABIDE2/Outputs/denoise`（每个 run 约 1 GB 的 4D 体数据） | — | 覆盖不完整或体量过大，DCAN 已能满足需要 |

示例文件核查（sub-<ID>，Pitt 站点）：ptseries 为 200 帧 × 379 个分区，TR 1.5 s，无 NaN、无零方差分区；时间序列**未剔帧**，剔帧由我们按掩膜执行。

**`desc-filtered` 的含义（2026-10-30 已核实，来源 abcd-hcp-pipeline.readthedocs.io 的首页与 Outputs 页）**：这是 DCANBOLDProc 处理之后的时间序列。处理内容：回归白质、脑脊液、全局信号均值，以及 6 个头动参数及其 Volterra 展开；回归系数只用好帧估计；然后用二阶 Butterworth 做 0.008–0.09 Hz 带通滤波，滤波前坏帧先用插值替换。剔帧掩膜采用 Power 2014 的规则：两段剔除帧之间不足 5 个连续好帧的也一并剔除。由此得出：
- 主管线**做了 GSR**，论文 Methods 必须写明；GSR 的影响由 B3 用 PCP 的有/无 GSR 版本评估。
- 时间序列已经带通滤波，**不计算 fALFF/ALFF**。
- ABIDE 这次运行时 DBP 去噪用的 FD 阈值和管线版本，文档没有写；文档中标准设置是 0.3 mm。需要查看衍生数据附带的 JSON 或日志，查不到就在局限中说明。

## 3. 纳入与排除

- 纳入：5–21 岁；dx_group = 1（ASD）或 2（TC）；表型表中有记录。
- 排除：ABIDEII-UCLA_Long、ABIDEII-UPSM_Long（纵向站点，与课题 A 一致）；DCAN 中缺少主图谱 ptseries 或剔帧掩膜者。
- 同一人有多个 session：取名称排序的第一个（B01 实测 18 人）。
- 符合条件的人数：表型层面为 1,717 人；有 DCAN 数据的最终人数以 `results/abide_dcan_download.json` 为准。

## 4. 软件与版本

课题 B 的 Python 脚本统一在项目虚拟环境 `.venv_B`（Python 3.12.10，继承系统已有的包）中运行，版本快照见 `docs_B/env_venv_B_20261030.txt`：nibabel 5.4.2，numpy 2.4.5，pandas 2.3.3，scipy 1.17.1，scikit-learn 1.9.1，nilearn 0.14.1，hcp-utils 0.1.0。R gamlss 5.5.5（课题 A）。abagen、neuromaps、brainspace 等在首次使用时安装，并重新导出快照。

分区表：`scripts/B03_parcel_table.py` → `docs_B/glasser_parcel_table.csv`。DCAN ptseries 的分区顺序是先右后左，hcp-utils 是先左后右，因此按名称对应；360 个皮层分区全部匹配，CAB-NP 12 个网络（加皮层下共 13 类）的归属来自 hcp-utils 的 `ca_network`。

## 5. 质控（不看诊断标签，预先写定）

| 步骤 | 规则 | 理由 |
|---|---|---|
| 剔帧 | **主分析 FD > 0.2 mm 的帧剔除**，直接使用 DCAN 掩膜中 FD_threshold = 0.20 一项（掩膜已包含其内置的邻帧剔除规则） | DCAN/ABCD 管线的常用设定；敏感性分析改用 0.3 mm |
| 剩余时长 | **主分析剔帧后剩余 ≥ 240 s**；如果按此标准合格的人数不足全部符合条件者的 60%，则改为 ≥ 180 s，这个决定只看合格率、不看诊断分组，并记录在第 12 节 | 平衡 FC 的信度与样本量；ABIDE 很多站点扫描仅 5–6 分钟 |
| 头动比较 | 报告剔帧后的平均 FD，以及 ASD 与 TC 的剩余帧数；所有组间模型都加入平均 FD 作为协变量 | ASD 组头动通常更大，是最主要的混杂 |
| 分区质量 | 被试层面：任一分区为 NaN 或零方差即排除；分区层面：如有分区在超过 10% 的被试中信号缺失，从所有分析中剔除该分区 | 皮层边缘或磁敏感伪影区 |
| 站点 | 质控后 TC < 10 人的站点不参与常模的站点重新居中，这些站点的被试只用于敏感性分析 | 站点校正需要足够的本地对照（课题 A 第 5 节） |
| 结构 | 皮层厚度沿用 DCAN 输出，FastSurfer 体积沿用课题 A 的质控结果 | — |

产物：`data/interim/abide_rest_qc_<日期>.csv`（每人：剩余帧数、剩余秒数、平均 FD、是否合格、排除原因），以及 CONSORT 式流程图（规则 3）。

## 6. 共用特征定义

| 特征 | 定义 | 课题 |
|---|---|---|
| 分区 FC 矩阵 | 剔帧后各分区时间序列两两 Pearson 相关，再做 Fisher z 变换；主图谱 HCP2016（379 × 379） | B1、B3、B4 |
| 分区 FC 强度 | 每个分区与其余分区 FC（z）的均值，只取正值连接（敏感性：取绝对值） | B1 |
| 网络间 FC | 按 Cole-Anticevic 网络划分（CAB-NP，12 个网络）求网络内和网络间 FC 均值；网络归属见 `docs_B/glasser_parcel_table.csv`（来源 hcp-utils 0.1.0） | B1、B3 |
| 皮层厚度 | DCAN fsLR32k 厚度（59,412 个皮层顶点），按 Glasser 360 分区求均值；顶点标签见 `docs_B/glasser_cortex_vertex_labels.npy`（hcp-utils `mmp.map_all` 的前 59,412 个点） | B1 |
| 皮层下体积 | 课题 A 的 FastSurfer 体积 | B1 |

## 7. 站点处理

与课题 A 一致：常模模型不含站点项；每个站点用本地 TC 对 z 值重新居中（k ≥ 10 时再按标准差缩放），TC 本人做重新居中时用留一法，避免自己参与自己的校正。敏感性分析：ComBat-GAM（保留年龄、性别和诊断的协变量效应）。B3 专门比较不同站点处理方案，不受本节约束。

## 8. 统计通用约定（规则 4）

- Seed：主分析 42；敏感性 7、123、2024。凡涉及随机过程（交叉验证、置换、spin test、bootstrap），所有 seed 都跑一遍，主要结果报告跨 seed 的均值 ± SD。
- 多重比较：分区层面用 BH-FDR（q < 0.05）；空间相关用 spin test（≥ 10,000 次置换）。
- 缺失：表型变量缺失超过 30% 的不作为模型变量，只在有该变量的子样本里做探索性分析，并说明子样本的代表性。ADOS、SRS 只在部分站点采集，各课题须报告可用人数。
- 发现与复现：ABIDE I 发现、ABIDE II 复现。复现标准在各课题 protocol 中预先写定（例如组水平偏离图的空间相关，并做 spin test）。

## 9. 阶段 0 的可行性检查（完成后冻结 v1.0）

| 检查 | 通过标准 | 状态 |
|---|---|---|
| DCAN 下载 | 符合条件者中 ≥ 90% 的 4 个文件齐全 | ✅ 1,621/1,717（94.4%），6,520 个文件全部校验通过（`results/abide_dcan_download.json`） |
| 质控后样本量 | ASD 和 TC 在 ABIDE I、II 中各 ≥ 200 人 | ⛔ **差 1 人**：ABIDE I ASD 225 / TC 307；ABIDE II ASD **199** / TC 254（`results/B02_qc_fc.json`）。剩余时长门槛按规则取 240 s（总体合格率 60.8% ≥ 60%）。处理方式待用户决定，见第 12 节 |
| 头动 | 报告组间平均 FD 差异，不设门槛，但必须报告 | ✅ 合格者保留帧的平均 FD：ASD 0.087 ± 0.017，TC 0.085 ± 0.018 mm；剩余时长 ASD 369 ± 131 s，TC 394 ± 148 s。**差异性排除**：合格率 ASD 54–55%，TC 63–71%，已在 B1 敏感性分析第 8 条中应对 |
| `desc-filtered` 含义 | 查到文档依据 | ✅ 见第 2 节 |
| FC 计算的独立复核 | 用不同实现复算 FC，最大差异 < 1e-4 | ✅ 随机 5 人（seed 42），最大差异约 6e-8（`scripts/B02v_verify_fc.py` → `results/B02v_verify_fc.json`） |
| Glasser fsLR32k 标签与 CAB-NP 网络划分 | 文件取得，分区顺序核对一致 | ✅ 2026-10-30，B03 断言通过（360/360） |
| FC 复现性的预检 | TC 组平均 FC 矩阵在 ABIDE I 与 II 之间的相关 r > 0.8（检查管线是否一致，不涉及诊断） | ✅ Glasser FD0.2 r = 0.987；FD0.3 r = 0.988；Gordon 0.983 / 0.985 |

## 10. 四个课题的边界（防切香肠）与启动门槛

| 课题 | 独立的科学问题 | 独占的分析 | 启动门槛 |
|---|---|---|---|
| **B1**（v0.2） | 个体 FC 常模偏离图和极端偏离的判定在分半、跨 run 之间有多可靠？只看信度达标的分区时，结构偏离与功能偏离是否对应？ | 个体常模 z 偏离的信度；结构–功能对应；校正后的异质性 | 阶段 0 通过即启动 |
| **B3** | ASD 静息态 FC 的结论在多大程度上取决于预处理、图谱和站点处理方式？ | PCP 16 种组合 × 图谱 × 站点处理；分类泄漏 | 阶段 0 通过即启动，与 B1 并行 |
| **B2** | 海马与杏仁核内部的动态共激活亚模式在 ASD 中是否改变？ | 种子内 μCAPs（需要 dtseries） | 先做预实验：μCAPs 在 TC 中的分半一致性（ABIDE I 内随机分半，空间相关 ≥ 0.7） |
| **B4** | ASD 的皮层层级（功能梯度、有向信号流）是否压缩？ | 扩散嵌入梯度、iEC | 先在扫描最长的站点做 iEC 重测预实验（同一人前后半段扫描的分区间 ICC ≥ 0.5）；不达标就只做梯度 |

共享的内容（样本、质控、FC 矩阵）在每篇中写成相同的 Methods 段落，并互相引用；结论不得重复：B1 不报告单纯的 ASD 与 TC 的 FC 差异（那是 B3 的内容），B3 不做常模和分子解释。

## 11. 目录与命名

- 原始衍生数据：`data/raw/abide_dcan/<cohort>/sub-<id>/<ses>/...`（ID 只用于本地处理，不进入论文、git 或 Zenodo，规则 3）
- 中间表：`data/interim/abide_<内容>_<YYYYMMDD>.csv`
- 结果：`results/B*_*.json`；脚本：`scripts/B0*_*.py`（共用）、`scripts/B1*_*.py`（B1）……
- 日志：`logs/B*_<YYYYMMDD>.log`

## 12. 变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-10-30 | v0.1 建立；主管线由原设想的 PCP 改为 DCAN | S3 核查发现 PCP 只有 ABIDE I，DCAN 同时覆盖 ABIDE I 与 II |
| 2026-10-30 | B1 的功能指标由 fALFF/ReHo 改为分区 FC 强度和网络间 FC | ptseries 是分区信号，无法计算 ReHo；之后核实 `desc-filtered` 已做 0.008–0.09 Hz 带通，fALFF 也无法计算 |
| 2026-10-30 | 被试层面的坏分区规则改为 Glasser 与 Gordon **两个图谱**都不能有 NaN 或零方差分区（原为只看主图谱） | 首次运行时有 Gordon 有坏分区、Glasser 正常的被试进入 Gordon 分析，产生 NaN；改为两图谱同一批人，保证主分析与图谱敏感性分析可比。此改动在看到任何诊断相关结果之前做出，共排除 21 人（`bad_parcels`） |
| 2026-10-30 | 可行性门槛"每组 ≥ 200"未通过：ABIDE II ASD 为 199 | 如实记录。**决定**：质控规则（FD 0.2 mm、≥ 240 s）不变，照常推进，论文局限中说明。理由：200 是可行性经验值，不是统计终点，差 1 人对统计效力影响可以忽略；看到人数后再改质控规则属于事后改动，代价更大。用户 2026-10-30 指示"先做只用 ABIDE 的项目"并推进 |
| 2026-10-30 | 课题 B 全部只用 ABIDE I+II，不纳入 HBN | 用户决定先做不依赖 HBN 的课题；B1 第 8 节中 HBN 作为第三队列的设想取消，如果将来加入，按新版本另行登记 |
