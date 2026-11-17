# 双副本 CPU 合成验收

在工程根目录用 Python 3.11+ 运行。每条命令使用新的输出目录。

```text
python -m replaylab preflight --config fixtures/synthetic/study.json --out out/m0/preflight
python -m replaylab run --config fixtures/synthetic/study.json --plan fixtures/synthetic/plan.json --out out/m0/run
python -m replaylab shadow --config fixtures/synthetic/study.json --manifest out/m0/run/manifest.json --out out/m0/shadow
python -m replaylab paired --config fixtures/synthetic/study.json --manifest out/m0/run/manifest.json --shadow out/m0/shadow/windows.csv --out out/m0/paired
python -m unittest discover -s tests -v
```

只运行双目录逐字节验收时使用 `python -m unittest discover -s tests -v -k test_t01`。测试用两套新目录、独立 Python 进程和不同 `PYTHONHASHSEED`，逐一比较文件清单、原始字节与 manifest 引用的文件摘要。夹具原始字节使用 UTF-8 与 LF，`expected/` 由测试读取。

合成 action 块中，全池 G 为 40、50、42，差值为 +10、+2、+8。同策略 A/A 使用冻结的合成时延噪声，差值为 -1。影子在途组拟启用的偏移为 0 ns，匹配 immediate；公共进展组的 50,000,000 ns 偏移未测，邻档为 0 和 100,000,000 ns。上述数值用于验证程序行为，研究状态留空。

虚拟时钟同刻按注入、反馈与终止、失败、锚点采样、启用、定时器、tick、到达、派发、传输交接排序。同一类别按入队顺序处理。新产生的反馈在产生它的网关操作之后入账。每轮均记录清理检查，诊断重跑独立保存在 `attempts.json`。真机执行在当前版本报告后端尚未接入，平台核验和设备时序由 M1 实测。
