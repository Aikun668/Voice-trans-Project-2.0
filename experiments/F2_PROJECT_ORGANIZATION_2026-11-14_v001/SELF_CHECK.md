# 规则对照

核验命令：项目根运行 `python -B experiments/F2_PROJECT_ORGANIZATION_2026-11-14_v001/verify.py`。实际运行结果见 verification.json。

| 规则 | 对应结果 | 证据 |
|---|---|---|
| 1 唯一现行路线 | 主项目ROADMAP为准，其他文件只导航；旧树明确历史 | 三处入口、FILE_INDEX、分区说明 |
| 2 角色/状态/来源和简短说明 | 四类、不同模型身份及源文件绑定已登记 | 分区说明、asset_roles.json、experiment_roles.csv |
| 3 先读再分、主次不颠倒 | 已实际读取指定对话，完成快照与实际报告相符 | feedback.md、ROADMAP §5、次线REPORT/verification |
| 4 两线状态区分 | 主线jade_rejected；次线0微调、门槛未过、pending_jade | ROADMAP、资产角色表；不冒称效果通过 |
| 5 17对和来源保留 | 34条绑定、14个源文件逐一匹配，未替换根路径 | source_bindings.csv、verification.json |
| 6 旧待办退出、原话保留 | 旧待办改去向表；历史正文和原话保留；同学任务未核实 | before/、H旧交接标记、日志和协作草案 |
| 7 旧实验及依赖保留 | 所有既有实验目录原位，四条已核对依赖登记 | inventory_before.json、retained_dependencies.json、verification.json |
| 8 快照和范围 | 18份文档快照，无提交/推送/安装/训练命令 | before_manifest.json、修改清单、本轮命令记录 |
| 9 链接/身份/保护核验 | 新链接无断链，源和资产哈希核验，旧原有断链单列 | verify.py、verification.json |
| 10 不设正式默认 | 旧指针原样保留，global_default_model=null | asset_roles.json、protected_before.json |
| 11 不新增听感判断 | 本轮未生成音频；明确Agent不能听音 | REPORT、分区说明与两线状态 |

这张表说明落实方式；量化核验以脚本本次实际输出为准，不把文档核验当成模型或听感验收。
