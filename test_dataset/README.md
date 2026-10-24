# 测试数据集说明

用于修音 PoC 的端到端测试。所有文件均为复制件，原始数据不动。

## 目录结构

```
test_dataset/
├── my_singing/          # 自己唱的（男声，手机录音），测试输入
├── web_singing/
│   └── gtsinger_alto_control/   # 中文歌正常唱法，修音目标参考
└── README.md
```

## my_singing（14 条，命名：歌曲_跑调档位.wav）

跑调档位：`normal`=正常 / `mild`=轻度 / `medium`=中度 / `severe`=严重

| 歌曲 | 档位 |
|---|---|
| long_low / long_high / scale（长音、音阶练习，含2个take2） | 全部 normal |
| molihua（茉莉花） | normal / medium / severe |
| shige（是谁在唱歌） | normal / medium / severe |
| zuixuan（最炫民族风） | normal / mild / severe |

注：mild 只有 zuixuan 一首；原始 m4a 见 `samples/raw`，登记表见 `samples/samples.csv`。

## web_singing

- **gtsinger_alto_control/**（15 条，命名：歌名_normal.wav）
  来源：GTSinger 数据集（HuggingFace），女中音 ZH-Alto-1 中文流行歌 Control_Group 片段，每首 6~15 秒。
  用途：专业"正常版"参考目标。
  注意：女声音区，比本人男声偏高；如需男声参考可补 ZH-Tenor-1。

## 用法约定

- my_singing 的 off 档位 → 输入修音链，量化修正前后音准误差
- gtsinger_alto_control → 当参考目标 / 试听对照
- 每条音频的音高标注（MIDI note）可从 GTSinger 官方 json 获取，本地子集见 `third_party/gtsinger_subset`
