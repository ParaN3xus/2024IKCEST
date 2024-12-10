#import "@preview/touying:0.5.3": *
#import "theme.typ": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/cetz:0.3.1"
#import "@preview/cetz-plot:0.1.0"
#import "@preview/codly:1.0.0": *
#import "data.typ": *

#set text(lang: "zh")
#set par(justify: true)
#let serif = ("Libertinus Serif", "Noto Serif CJK SC")
#let sans = ("Libertinus Sans", "Noto Sans CJK SC")
#let mono = ("DejaVu Sans Mono", "Noto Sans CJK SC")

#let offset(anchor, x, y) = {
  (rel: (x, y), to: anchor)
}

#show <numbering-keep>: it => {
  counter(heading).update((..x) => {
    let (fst, sec, ..more) = x.pos()
    (fst, sec - 1, ..more)
  })
  it
}
#show <not-outlined>: set heading(outlined: false)
#show footnote.entry: set text(size: 12pt)

#show: codly-init.with()

#show: my-university-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: text(size: 28pt)[
      2024 IKCEST第六届"一带一路"国际大数据竞赛 \
      暨第十届百度&西安交大大数据竞赛
    ],
    title-footer: "2024 IKCEST第六届\"一带一路\"国际大数据竞赛暨第十届百度&西安交大大数据竞赛",
    subtitle: text(font: serif, size: 24pt, "AI体育赛事解说"),
    author: [
      #set text(size: 16pt)
      xyzw队 \ 复赛排名 nth
    ],
    author-footer: "xyzw队",
    date: datetime.today(),
    logo: box(
      baseline: 25%,
      {
        image("assets/masked.png", width: 40pt)
      },
    ),
  ),
  config-page(margin: (top: 3.5em, x: 2em)),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#set text(font: serif)
#title-slide()


== 团队成员 <touying:hidden>


#let members = (
  "xyz": (
    "img": image("assets/heads/masked.png", height: 10em),
    "desc": [
      xxxx大学 \
      aaaa学院 \
      1111专业 \
      xyz生
    ],
  ),
  "abc": (
    "img": image("assets/heads/masked.png", height: 10em),
    "desc": [
      yyyy大学 \
      bbbb学院 \
      2222专业 \
      abc生
    ],
  ),
  "123": (
    "img": image("assets/heads/masked.png", height: 10em),
    "desc": [
      zzzz大学 \
      cccc学院 \
      3333专业 \
      123生
    ],
  ),
  "tuv": (
    "img": image("assets/heads/masked.png", height: 10em),
    "desc": [
      wwww大学 \
      dddd学院 \
      4444学 \
      tuv生
    ],
  ),
)


#align(
  center + horizon,
  grid(
    columns: (1fr,) * members.keys().len(),
    row-gutter: 1em,
    align: center + horizon,
    ..members.keys().map(k => members.at(k).img),
    ..members
      .keys()
      .map(k => {
        set text(size: 17pt)
        k
        linebreak()
        members.at(k).desc
      }),
  ),
)

== 目录 <touying:hidden>

#outline(depth: 2, indent: 2em)


= 赛题解析

== 赛题解析 <not-outlined>

- 背景: AI体育赛事解说
- 任务描述: 本次比赛旨在通过体育比赛视频, 挑战参赛者使用计算机视觉技术对*动态画面进行深入分析*, 结合*大语言模型*、自然语言处理、语音生成等技术, 通过大模型智能体/插件设计生成赛事解说. 通过这一技术挑战, 参赛者将有机会展示他们在*视频处理、目标识别和跟踪、大语言模型*等方面的技能, 推动体育视频分析技术的进步.

#pause

#cetz.canvas({
  import cetz.draw: *
  content(
    (0, 0),
    image(
      "assets/tbd/vid.png",
      height: 5em,
    ),
    name: "vid",
  )

  line(
    offset("vid.east", 1, 0),
    offset("vid.east", 3, 0),
    mark: (end: ">", fill: black),
    name: "arr1",
  )

  content(
    offset("arr1.end", 5.7, 0),
    image(
      "assets/intro/multitarget-unlabeled.png",
      height: 5em,
    ),
    name: "track",
  )

  line(
    offset("track.east", 1, 0),
    offset("track.east", 2, 0),
    offset("track.east", 2, -1.6),
    mark: (end: ">", fill: black),
    name: "arr2",
  )

  content(
    offset("arr2.end", 0, -2.5),
    image(
      "assets/intro/multitarget-labeled.png",
      height: 5em,
    ),
    name: "label",
  )

  line(
    offset("label.west", -1, 0),
    offset("label.west", -3, 0),
    mark: (end: ">", fill: black),
    name: "arr3",
  )

  content(
    offset("arr3.end", -2.5, 0),
    [
      #box(width: 5em)[
        #set text(size: 7.5pt)
        ```json
        {
          "events": [
            {
              "type": "pass",
              ...
            },
            ...
          ]
        }
        ```
      ]
    ],
    name: "event",
  )

  line(
    offset("event.west", -1, 0),
    offset("event.west", -3, 0),
    mark: (end: ">", fill: black),
    name: "arr4",
  )

  content(
    offset("arr4.end", -2.8, 0),
    [
      #box(width: 6em)[
        #set text(size: 10pt)
        各位观众大家好, 您现在看到的是世界杯四分之一决赛, 对阵双方是...
      ]
    ],
    name: "event",
  )
})


= 技术路线

== 多目标追踪 <numbering-keep>

使用常见的TbD(Tracking-by-Detection)方法.

#align(
  center,
  cetz.canvas(
    length: 30pt,
    {
      import cetz.draw: *

      content(
        (0, 0),
        image(
          "assets/tbd/vid.png",
          width: 8em,
        ),
        name: "vid",
      )

      line(
        offset("vid.east", 1, 0),
        offset("vid.east", 3, 0),
        mark: (end: ">", fill: black),
        name: "arr1",
      )

      content(
        offset("arr1.mid", 0, 0.5),
        [
          #set text(size: 14pt)
          帧提取
        ],
      )

      content(
        offset("arr1.end", 3.3, 0),
        image(
          "assets/tbd/frames.png",
          width: 7em,
        ),
        name: "frames",
      )

      line(
        offset("frames.south", 0, -0.1),
        offset("frames.south", 0, -1.43),
        mark: (end: ">", fill: black),
        name: "arr2",
      )

      content(
        offset("arr2.mid", -1.3, 0),
        [
          #set text(size: 14pt)
          目标检测
        ],
      )

      content(
        offset("arr2.end", 0, -2.2),
        image(
          "assets/tbd/tracked_frames.png",
          width: 7em,
        ),
        name: "tracked_frames",
      )

      rect(
        offset("tracked_frames", 1, -0.4),
        offset("tracked_frames", 1.5, 0.1),
        name: "crop_rect",
      )

      content(
        offset("arr2.mid", 8, 0),
        image(
          "assets/tbd/cropped.png",
          width: 8em,
        ),
        name: "crop_content",
      )

      line("crop_rect.south-east", "crop_content.south-east")
      line("crop_rect.south-west", "crop_content.south-west")
      line("crop_rect.north-east", "crop_content.north-east")
      line("crop_rect.north-west", "crop_content.north-west")

      line(
        ("tracked_frames", "-|", "arr1.end"),
        ("tracked_frames", "-|", "arr1.start"),
        mark: (end: ">", fill: black),
        name: "arr3",
      )

      content(
        offset("arr3.mid", 0, 0.5),
        [
          #set text(size: 14pt)
          目标追踪
        ],
      )

      content(
        ("arr3.mid", "-|", "vid"),
        image(
          "assets/tbd/tracked_vid.png",
          width: 8em,
        ),
        name: "crop_content",
      )
    },
  ),
)

#heading(depth: 2, outlined: false, "多目标追踪: 目标识别") <numbering-keep>


检测模型: YOLOX #footnote[https://github.com/Megvii-BaseDetection/YOLOX]-s 预训练模型 + 初赛数据微调

#{
  let h = 7
  let w = 10.6
  grid(
    columns: (1fr, 1fr),
    cetz.canvas({
      import cetz-plot: *
      plot.plot(
        size: (w, h),
        x-label: "Epoch",
        y-label: "Test Loss",
        x-tick-step: 50,
        {
          plot.add(
            style: (stroke: 1.3pt + blue),
            range(300).zip(loss).slice(0, 256),
          )
        },
      )
    }),
    cetz.canvas({
      import cetz-plot: *
      plot.plot(
        size: (w, h),
        x-label: "Epoch",
        y-label: "Value",
        legend: (6.4, 3.5),
        x-tick-step: 50,
        {
          plot.add(
            style: (stroke: 1.3pt + blue),
            label: "person acc",
            range(300)
              .zip(acc1.map(x => x * 1.05))
              .slice(0, 256)
              .enumerate()
              .filter(x => calc.rem(x.first(), 5) == 0)
              .map(x => x.last()),
          )

          plot.add(
            style: (stroke: 1.3pt + yellow),
            label: "ball acc",
            range(300).zip(acc2).slice(0, 256).enumerate().filter(x => calc.rem(x.first(), 5) == 0).map(x => x.last()),
          )

          plot.add(
            style: (stroke: 1.3pt + green),
            label: "person rec",
            range(300).zip(rec1).slice(0, 256).enumerate().filter(x => calc.rem(x.first(), 5) == 0).map(x => x.last()),
          )

          plot.add(
            style: (stroke: 1.3pt + red),
            label: "ball rec",
            range(300).zip(rec2).slice(0, 256).enumerate().filter(x => calc.rem(x.first(), 5) == 0).map(x => x.last()),
          )
        },
      )
    }),
  )
}

在 166 Epoch 上获得了较好的结果, 球和人的识别准确率都超过 90%.

#heading(depth: 2, outlined: false, "多目标追踪: 目标追踪") <numbering-keep>

在线追踪器, 通过*匈牙利算法*把新的帧中的目标匹配到最好的序列.

#align(
  center,
  cetz.canvas(
    length: 26pt,
    {
      import cetz.draw: *

      let l = 3

      content((0, 0), box(inset: 8pt, "代价"), name: "price")

      line(
        "price.north-east",
        offset((0, 0), l, l),
        mark: (end: ">", fill: black),
        name: "arr1",
      )
      line(
        "price.south-east",
        offset((0, 0), l, -l),
        mark: (end: ">", fill: black),
        name: "arr2",
      )

      content(
        offset("arr1.end", 7.74, 0),
        [
          #std.grid(
            columns: 3,
            gutter: 1em,
            align: center + horizon,
            "外观",
            {
              image(
                "assets/track/appearance.png",
                height: 8em,
              )
              set text(size: 9pt)
              place(
                left + top,
                dx: 5pt,
                dy: 15pt,
                {
                  set text(rgb("D8481E"))
                  $bold(o_1) = {0.283, 0.461, 0.389, ..., 0.314}$
                },
              )
              place(
                left + bottom,
                dx: 5pt,
                dy: -6pt,
                {
                  set text(rgb("AB1FD6"))
                  $bold(o_2) = {0.708, 1.997, 0.592, ..., 0.933}$
                },
              )
            },
            {
              image(
                "assets/track/vectors.png",
                width: 8em,
              )
              place(
                right + top,
                dx: -35pt,
                dy: 45pt,
                {
                  set text(rgb("D8481E"), size: 14pt)
                  $bold(o_1)$
                },
              )
              place(
                left + top,
                dx: 32pt,
                dy: 70pt,
                {
                  set text(rgb("AB1FD6"), size: 14pt)
                  $bold(o_2)$
                },
              )
              place(
                center + top,
                dx: -3pt,
                dy: 65pt,
                {
                  set text(size: 14pt)
                  $theta$
                },
              )
            },
          )
        ],
      )


      content(
        offset("arr2.end", 6, 0),
        [
          #std.grid(
            columns: 2,
            gutter: 1em,
            align: center + horizon,
            "运动",
            image(
              "assets/track/motion.png",
              height: 8em,
            ),
          )
        ],
      )
    },
  ),
)

#heading(depth: 2, outlined: false, "多目标追踪: 目标追踪") <numbering-keep>

- 几乎所有目标都在运动
- 方向改变多
  - 摄像头移动
  - 球员/球自身的方向改变

#pause

我们使用了一个很简单的*线性模型*, 通过序列中已有的的 BBox 值来预测下一帧的 BBox 值, 并通过计算预测值和下一帧中所有目标值的 IoU 来判断吻合度.

#pause

#grid(
  columns: 2,
  column-gutter: 2em,
  cetz.canvas({
    import cetz-plot: *
    plot.plot(
      size: (11, 5),
      y-min: 0.7,
      y-max: 0.92,
      y-label: "Average IoU",
      x-label: "Window size",
      y-tick-step: 0.025,
      {
        plot.add-bar(
          bar-width: .5,
          range(1, 11).zip(ious),
        )
      },
    )
  }),
  [
    #align(horizon)[
      经过实验: \
      取前三帧(即 Window size = 3)时, 预测 BBox 和实际值的 Average IoU 最好, 达到了 89.44%.
      #v(1.7em)
    ]
  ],
)

#heading(depth: 2, outlined: false, "多目标追踪: 目标追踪") <numbering-keep>


- 外观特征很相似(尤其是同队成员)
- 在初赛数据集上训练比较困难: 帧与帧之间差别小

#pause

直接使用了在 Market1501 数据集上训练的 ReID 模型, 并通过正规化使模型可以应用在我们的任务上.

#align(
  horizon + center,
  [
    #set text(size: 22pt)
    #table(
      columns: 3,
      inset: 1em,
      [Metrics], [R1], [mAP],
      [Value], [0.5912], [0.2282],
    )
  ],
)

== 球衣号识别


整体思路参考 jersey-number-pipeline.
#pause

- 球的标签是 -1. #pause
- 部分识别失败的原因
  - 部分本来能看清球衣号的图片被当作离群点/不可读筛除
  - 姿态识别不够准确, 球衣号被截断


== 比赛事件重建


#figure(
  grid(
    columns: 3,
    gutter: 1em,
    figure(image("assets/rebuild/head.png", width: 90%, height: 85%), numbering: none, caption: [a) 头球]),
    figure(image("assets/rebuild/own.png", width: 90%, height: 85%), numbering: none, caption: [b) 拥有球]),
    figure(image("assets/rebuild/fall.png", width: 90%, height: 85%), numbering: none, caption: [c) 摔倒]),
  ),
)

#figure(
  image("assets/rebuild/fakeball.png"),
)

#figure(
  image("assets/rebuild/frames.png", width: 120%),
)

#figure(
  image("assets/rebuild/intense.png"),
)

#figure(
  image("assets/rebuild/ballkeeper.png"),
)

== 智能体解说

#cetz.canvas({
  import cetz.draw: *
  content(
    (0, 0),
    [
      #box(width: 10em)[
        #set text(size: 12.5pt)
        ```json
        {
          "events": [
            ...
          ]
          "match_info": [
            ...
          ]
        }
        ```
      ]
    ],
    name: "output",
  )

  line(
    offset("output.east", 1, 0),
    offset("output.east", 3, 0),
    mark: (end: ">", fill: black),
    name: "arr1",
  )

  content(
    offset("arr1.end", 5.7, 0),
    [
      #box(width: 10em)[
        #set text(size: 12.5pt)
        #show raw: it => {
          show regex("\p{script=Han}"): set text(font: "Noto Sans CJK SC")
          it
        }
        ```Prompt
        你是...
        我会给你...
        你的任务是...
        你要遵循...
        你不可以做...
        ```
      ]
    ],
    name: "prompt",
  )

  line(
    offset("prompt.east", 1, 0),
    offset("prompt.east", 3, 0),
    offset("prompt.east", 3., -2.5),
    mark: (end: ">", fill: black),
    name: "arr2",
  )

  content(
    offset("arr2.end", 0, -3.5),
    image(
      "assets/rebuild/ernie-logo.png",
      height: 5em,
    ),
    name: "ernie",
  )

  line(
    offset("ernie.west", -1, 0),
    offset("ernie.west", -3, 0),
    mark: (end: ">", fill: black),
    name: "arr3",
  )

  content(
    offset("arr3.end", -3, 0),
    image(
      "assets/rebuild/user-input.png",
      width: 7em,
      height: 7em,
    ),
    name: "event1",
  )

  line(
    offset("event1.west", -1, 0),
    offset("event1.west", -3, 0),
    mark: (end: ">", fill: black),
    name: "arr4",
  )

  content(
    offset("arr4.end", -2.8, 0),
    [
      #box(width: 6em)[
        #set text(size: 10pt)
        各位观众大家好, 您现在看到的是世界杯四分之一决赛, 对阵双方是...
      ]
    ],
    name: "event1",
  )
})

#pagebreak()

#{
  let h = 10
  let w = 24.5
  let points = ((9, 4.85), (11, 7), (27, 17.24), (46, 32)).map(((x, y)) => {
    (x, y)
  })
  cetz.canvas({
    import cetz-plot: *
    plot.plot(
      size: (w, h),
      x-label: "视频时长 (s)",
      y-label: "算法均处理耗时 (min)",
      y-min: 0,
      x-tick-step: 5,
      {
        plot.add(
          mark: "x",
          points,
          line: "spline",
        )
      },
    )
  })
}

#align(center + horizon)[
  #image("assets/rebuild/trick-flow.png")
]

#pagebreak()


- 幻觉?
- 让智能体听话

= 总结展望


== 总结

- 通过多目标追踪, 球衣号识别获得了场上球&人的运动轨迹
- 通过上述运动轨迹和其他信息实现了球赛事件重建
- 基于重建的事件, 通过 LLM 生成解说词

== 展望

#pause
+ 我们的作品 #pause
  - 整体的事件 #pause
  - 提示词 #pause
    - 事件->话术 #pause
    - 考虑实际情景: 解说 #pause
+ 足球赛事解说 #pause
  - CV? 原始数据? #pause
  - 更好的LLM


#let typst = text(
  size: 1.1em,
  font: "Buenard",
  baseline: -0.0em,
  fill: rgb("#a3b4cc"),
  "typst",
)

#focus-slide[

  #align(center + horizon)[
    \#thanks
  ]

  #place(bottom + center)[
    #set text(size: 18pt)

    #grid(
      columns: 3,
      [Slides generated by #typst with package Touying],
    )
  ]
]

