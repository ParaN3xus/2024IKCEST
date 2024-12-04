#import "@preview/cheq:0.1.0": checklist

#show: checklist

#set text(lang: "zh", font: ("Libertinus Serif", "Noto Sans CJK SC"))

#set page(numbering: (..x) => context { str(x.pos().first()) + "/" + str(counter(page).final().first()) })

#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  set text(size: 20pt)
  align(center, it)
}

#show link: set text(fill: blue.lighten(20%))
#show link: underline
#show table: align.with(center)



#let work(..names) = {
  text(fill: orange)[(#names.pos().join(" & ") working on)]
}
#let done(..names) = {
  text(fill: green.darken(10%))[(✓ by #names.pos().join(" & "))]
  for name in names.pos() {
    counter("_complete_task_" + name).step()
  }
}

#let waiting = {
  text(fill: red, weight: "bold")[WAITING]
}

#align(center)[#text(size: 24pt, weight: "bold")[IKCEST2024 我要跟你一队]]

= Schedule
#let names = ("a", "b", "c", "d")

#table(
  columns: names.len() + 1,
  align: center + horizon,
  [Name], ..names,
  [Complete tasks], ..names.map(x => context counter("_complete_task_" + x).final().at(0))
)

- [x] #done("a", "d") 过初赛
  - 参考
    - #link("https://github.com/luanshiyinyang/awesome-multiple-object-tracking/blob/master/README-zh.md")[Awesome] 目录
    - 数据集: #link("https://github.com/SoccerNet/sn-tracking")[sn-tracking]
    - MOT 基本结构介绍: #link("https://mmtracking.readthedocs.io/en/latest/tutorials/customize_mot_model.html")[Docs]

  - [x] #done("a") 交个baseline拿排位先
  - [x] #done("c") 研究一下评分标准
  - [x] #done("a", "d") 看看GHOST好不好用

    GHOST 不管检测, 只管追踪, 所以需要yolox来做目标检测
    - [x] #done("a") 用yolox做目标检测, 然后给GHOST的detections_GHOST
    - [x] #done("a") 用GHOST生成追踪
      - [x] #done("d") 写一个脚本把所有后缀不是.txt的文件改成.txt
    - [x] #done("a") 交结果 *RK17*
  - [x] #done("a") 可能其他方式提高初赛排名, 如超参数优化, 或者尝试其他模型
    - [x] #done("a") 在不区分不同球员的数据上微调yolox, 解决球的识别, 并提高人的识别精度
      - [x] #done("a") 修改转数据集的脚本, 调整分类和分割
      - [x] #done("a") 训练 *RK12*
      - [x] #done("a") 使用最新训练的yolox, 加上垃圾filter *RK11*
    - [ ] #strike[#work("c") 使用 C-BIoU 作为追踪器, 改善掉追踪换id问题]: 他摆烂了
      - [ ] #strike[#work("c") 实现 C-BIoU]
        - 参考:
          - #link("https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Hard_To_Track_Objects_With_Irregular_Motions_and_Similar_Appearances_WACV_2023_paper.pdf")[C-BIoU 原文]
          - IMM Estimator #link("https://filterpy.readthedocs.io/en/latest/kalman/IMMEstimator.html")[Docs] #link("https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V22-N04/22-04-Genovese.pdf")[原论文]
          - SORT #link("https://github.com/typst/packages/pull/954")[GitHub]
          - IoU #link("https://github.com/bochinski/iou-tracker")[GitHub]

    - [ ] 在我们的数据集上微调GHOST, 达到更好的追踪效果

- [x] #done("a", "b", "d", "c") 过复赛 *RK6*
  - [ ] #strike[需要用新的数据微调yolox, 以更好的识别不同的球员和球]
    - [x] #done("a") 把数据集转成coco格式
    - [x] #done("a") 完成训练用的设施
    - [x] #done("b")有好几场比赛, 每场比赛衣服颜色都不一样, 妈的: 结论: 对球衣 OCR 获得球员号
      - [x] #done("b")验证测试集中的比赛是不是在训练集中都有出现
        - [x] #done("b")如果是, 那就把数据集再分为不同比赛场次的N份
        - [x] #done("b")如果不是, 那就把训练集按照衣服颜色分类, 让类别尽量少, 只要不出现同一数据集中左队和右队衣服颜色会混淆即可
    - [ ] #strike[训练]
  - [ ] #strike[用重新训练的数据生成detections, 然后给GHOST识别]
  - 参考:
    - #link("https://www.ikcest.org/bigdata2024/spzs/content/240935270310426722.html")[官方赛题解说]
    - #link("https://aistudio.baidu.com/datasetdetail/291145")[复赛数据集下载]
  - [x] #done("a")OCR部分
    - 参考: #link("https://github.com/mkoshkina/jersey-number-pipeline")[mkoshkina/jersey-number-pipeline]
    - [x] #done("a")跑通 jersey-number-pipeline *RK7*
    - [x] #done("a") 结合 YOLOX 的足球识别结果
    - [ ] #strike[在我们的数据上训练]: 训练不了, 他给的数据只有最后结果, 我们还需要很多过程量
  - [x] #done("a") 部署和算法部分的运行文档
  - [x] #done("b", "a", "d", "c") 智能体插件部分
    - 参考:
      - #link("https://github.com/yHan234/wx-yolo")[智能体示例]
      - #link("https://pan.baidu.com/s/1b32T_hO0ubgm-S-mVy8BMA?pwd=eg2y")[智能体官网网页视频解析脚本]
    - [x] #done("b") 下载视频
    - [x] #done("b") 切片
    - [x] #done("b") yolox检测
    - [x] #done("b") GHOST(?) 跟踪
    - [x] #done("b") OCR
    - [x] #done("b", "a") 分析出球员所属球队, 是否为守门员\
      - [x] #done("d") 去除他妈的草地
    - [x] #done("b") 识别比赛信息: 时间, 参赛队伍, 比分
    - [x] #done("b") 解析OCR结果, 判断比赛事件
    - [x] #done("b") 综合信息, 输出prompt
    - [x] #done("b", "c", "a") prompt engineering
  - [x] #done("a", "b") 部署
- [x] #done("a", "b") 决赛
  - [x] qa
    - 复赛成绩明细: 0.3算法 + 0.5智能体 60.78 工作流编排合理 时间可以 有结果 不排除有幻觉 提前一天 实现 不要作弊 体现技术思考和*创新*能力 优化空间
    - 能不能看speakers notes: 能
    - 看到比赛页面里面说要展示解说语音并与视频合并效果: 不用
  - [x] #done("a", "b") 做PPT
  - [x] #done("a", "b")答辩
- [x] #done("a", "b", "d", "c") Excitement guaranteed *RK7*



= 算法部分 Pipeline
== YOLOX 目标检测
=== 所需设施
- #link("https://github.com/Megvii-BaseDetection/YOLOX")[YOLOX]
- #link("https://github.com/ParaN3xus/2024IKCEST/raw/refs/heads/main/YOLOX/get_dets.py")[`YOLOX/get_dets.py`]
- #link("https://github.com/ParaN3xus/2024IKCEST/releases/download/best_ckpt.pth/best_ckpt.pth")[`YOLOX/best_ckpt.pth`]
- #link("https://github.com/ParaN3xus/2024IKCEST/raw/refs/heads/main/YOLOX/my.py")[`YOLOX/exps/example/custom/my.py`]
- 环境:
  - python: `Python 3.8.19.`
  - pytorch: `py3.8_cuda11.1_cudnn8.0.5_0` installed with
    ```sh
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```

=== 输入
把输入的视频抽帧, 在 `YOLOX/datasets/` 下, 按如下结构组织
- MOT20/
  - test/
    - SNMOT-001/
      - seqinfo.ini
      - img1/
        - 000001.jpg
        - 000002.jpg
        - 000003.jpg
        ...
        - 000750.jpg
    - SNMOT-002/
      - seqinfo.ini
      - img1/
        - 000001.jpg
        - 000002.jpg
        - 000003.jpg
        ...
        - 000750.jpg
    ...

其中 `seqinfo.ini` 格式如下
```ini
[Sequence]
name=SNMOT-116
imDir=img1
frameRate=25
seqLength=750
imWidth=1920
imHeight=1080
imExt=.jpg
```

然后在 `YOLOX/` 下, 执行命令
```sh


```

=== 输出
该命令将会在 `YOLOX/datasets/MOT20_dets1/test` 下, 对每个 `SNMOT-xxx` 生成一个同名的文件夹, 如
- SNMOT-001/
  - det/
    - yolox_dets.txt
- SNMOT-002/
  - img1/
    - yolox_dets.txt
...

== GHOST 目标追踪
=== 所需设施
- #link("https://github.com/dvl-tum/GHOST")[GHOST]
- 用仓库的GHOST文件夹覆盖
- 环境: 复用YOLOX环境即可

=== 输入
把 `GHOST/datasets/detections_GHOST/MOT20/test/` 链接到 YOLOX 输出的 test 文件夹.
把 `GHOST/datasets/MOT20/test/` 链接到 YOLOX 的输入 `test` 文件夹.

然后在 `GHOST/` 下, 执行命令
```sh
bash scripts/main_20.sh
```

GHOST 的输出没有后缀名, 手动加上: 在 `GHOST/` 下, 执行命令
```sh
python tools/rename.py
```

GHOST 的输出出于某种未知原因, 比例正好小了 3 倍, 所以需要手动缩放回来: 在 `GHOST/` 下, 执行命令
```sh
python tools/rescale.py
```

=== 输出
将会在 `GHOST/out/scaled/` 下生成若干个文件名同上述 `SNMOT-xxx` 文件夹名的文本文件, 末尾有 `.txt` 后缀.

== jersey-number-pipeline 球衣号识别
=== 所需设施
- #link("https://github.com/mkoshkina/jersey-number-pipeline")[jersey-number-pipeline]
- 环境复用 YOLOX 的, 但是安装这个他会给你自动创建好几个conda环境

=== 输入
上述 GHOST 的输出中, 每个 .txt 文件为 csv 格式, 其中包含若干行, 每行所含信息如下
```
frame, track_id, left, top, width, height, -1.0, -1.0, -1.0, -1.0
```
除了前两个为 int, 其他都是 double. 每个 track_id 对应一个物体.

在 `jersey-number-pipeline/data/SoccerNet/test/images` 下, 应该有若干名为 SNMOT-xxx_yyy 的文件夹, 其中 xxx 为 SNMOT-xxx 的 xxx, yyy 为 track_id, 每个 SNMOT-xxx_yyy 文件夹内, 要把对应 track_id 在对应 SNMOT-xxx 中所有出现时的区域裁切出来, 保存到名为 foldername_no.jpg 的图片中, 例如
- jersey-number-pipeline/data/SoccerNet/test/images
  - SNMOT-001_1/
    - SNMOT-001_1_1.jpg
    - SNMOT-001_1_2.jpg
    - SNMOT-001_1_3.jpg
  - SNMOT-001_2/
    - SNMOT-001_2_1.jpg
    - SNMOT-001_2_2.jpg
    - SNMOT-001_2_3.jpg
  - SNMOT-002_1/
    - SNMOT-002_1_1.jpg
    - SNMOT-002_1_2.jpg
    - SNMOT-002_1_3.jpg
  ...

在 `jersey-number-pipeline/` 下执行:

/*
在 `jersey-number-pipeline/data/SoccerNet` 下, 应该有若干名为 SNMOT-xxx 的文件夹, 每个文件夹有一个 images 文件夹, images 文件夹内包含若干个名字为 track_id 的文件夹, 每个 track_id 文件夹内, 要把对应 track_id 在所有帧出现时的区域裁切出来, 保存到名为 track_id_no.jpg 的图片中, 例如
- jersey-number-pipeline/data/SoccerNet/
  - SNMOT-001/
    - images/
      - 1
        - 1_1.jpg
        - 1_2.jpg
        - 1_3.jpg
      - 2
        - 2_1.jpg
        - 2_2.jpg
        - 2_3.jpg
      - 3
        - 3_1.jpg
        - 3_2.jpg
        - 3_3.jpg
      ...
  - SNMOT-002/
  ...

注意 jersey-number-pipeline 每次只能处理一个 SNMOT-xxx 文件夹, 所以针对所有以上文件夹, 要循环以下步骤.

把要处理的那个 `SNMOT-xxx` 文件夹在原位置重命名为 `test`, 在 `jersey-number-pipeline/` 下执行:
*/

```py
python3 main.py SoccerNet test
```

=== 输出

以上命令会在 `jersey-number-pipeline/out/SoccerNetResults` 下生成 `final_results.json`, 存储了每个 track_id (文件夹名) 对应的球衣号, 如
```json
{
    "SNMOT-001_1": -1,
    "SNMOT-001_2": 9,
    "SNMOT-002_1": 8,
    "SNMOT-003_1": 1,
}
```
无法判断的文件夹对应的球衣号为 -1.

此外, 以上命令还会在 `jersey-number-pipeline/out/SoccerNetResults` 下生成 `soccer_ball.json`, 这里面存储了怀疑为足球的 track_id (虽然但是, YOLOX 也能拿到这个信息), 由于每个 SNMOT-xxx 都可能包含一个足球(或者多个)的 track_id, 所以这里面也有多个, 格式为
```json
{
    "ball_tracks": [
        "SNMOT-003_1",
        "SNMOT-004_5",
        "SNMOT-005_2"
    ]
  }
```
这里面的 track_id 同时也会在 `final_results.json` 中出现, 而且对应的球衣号固定为 1, 需要注意筛除.

== 比赛事件重建
通过上述步骤拿到 track_id 对应球衣号. 然后可以根据目标追踪的 bbox 得知球员的运动轨迹.

=== 带球
足球和某球员距离近?

=== 传球
带球人发生改变?

=== 进球
足球方向突变?

=== 没进球
idk

= 评价指标

参考: #link("https://motchallenge.net/results/MOTSynth-MOTS/")[MOT Benchmark] #link("https://blog.csdn.net/fuss1207/article/details/126806570")[CSDN 文章]

== $"MOTA"$

$
  "MOTA" = 1 - ("FN" + "FP" + Phi) / T
$

其中:

- $Phi$ (Fragmentation): 跟一半变 id 的次数
- $"FN"$ (False Negative): 所有帧里漏框的总数
- $"FP"$ (False Positive): 所有帧里多框的总数
- $T$ (True Positive): 所有帧里框对的总数

== $"IDF"_1$

$
  "IDF"_1 = (2 "IDTP" dot "IDFP") / ("IDTP" + "IDFP")
$

即 $"IDTP"$ 和 $"IDFP"$ 两个指标的调和平均, (存疑, 还没看, 原论文: #link("https://arxiv.org/abs/1609.01775")[arXiv])

== $"HOTA"$



= 模型数据 Pipeline

#let step_info = ("Step", "Desc.", "Progress", "src", "Assignee")

#let steps = (
  (
    "1",
    "YoloX",
    [],
    [
      #link("https://github.com/Megvii-BaseDetection/YOLOX")[GitHub]
    ],
    [],
  ),
  (
    "2",
    "GHOST",
    [],
    [
      #link("https://github.com/dvl-tum/GHOST")[GitHub]
      #link("https://arxiv.org/abs/2206.04656")[arXiv]
    ],
    [],
  ),
)

#table(
  columns: step_info.len(),
  table.header(..step_info),
  ..steps.flatten()
)
