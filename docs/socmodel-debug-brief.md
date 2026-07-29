# socModel 0 / CREATE_DEVICE 失败 — 根因与修复（已解决）

**结论：不是 SDK 缺陷，是我们把 `soc_model` 填错了。**
`soc_model` 必须是 `Qnn_SocModel_t` 枚举值 **87**，我们填的 **660** 是 Android 的
`soc_id`。两个数字都"属于 SM8850"，但属于两套完全不同的命名空间。

## 根因

`include/QNN/QnnTypes.h:1889`：

```c
QNN_SOC_MODEL_SM8850 = 87,
```

而 660 来自 `qti/aisw/tools/core/utilities/devices/android/android_device_constants.py:94`：

```python
SOC_ID_TO_SOC_NAME = { ... "660": "SM8850", ... }   # /sys/devices/soc0/soc_id
```

`CompileConfig` 把 `soc_details` 里的 `soc_model` 原样写进 backend extension 的
device custom config。`libQnnHtp.so` 在 `createDeviceHandle()` 里按
`Qnn_SocModel_t` 解析，查不到 660 → 回落为 0 → 报
`Unknown config socModel 0` → `CREATE_DEVICE: 11`。

原始 brief 里"Python 层配置正确"这条判断是错的：配置**成功送达**了 extension，
但送达的**值本身**无效。传递链没有断，从头到尾都完好。

### 为什么显式写 `soc_model` 反而绕过了 SDK 的自愈

`qairt/api/compiler/config.py:255`：

```python
if not soc_details.model or not soc_details.dsp_arch:
    if not populate_soc_details_from_factory(soc_details):
```

工厂查表**只在 model 或 dsp_arch 缺失时**触发。我们同时显式写了
`dsp_arch:v81` 和 `soc_model:660`，于是查表被跳过，错误值直通到底。
只写 `chipset:SM8850` 反而会被 SDK 自动解析成正确的 87。

## 验证证据

SDK 自身的查表（`DeviceFactory.get_device_soc_details("HTP", ...)`）：

| 输入 | chipset | model | dsp_arch | vtcm | hvx | fp16 |
|---|---|---|---|---|---|---|
| `"SM8850"` | SM8850 | **87** | 81 | 8MB | 8 | True |
| `"660"` | UNKNOWN_SDM | 0 | 0 | 0 | 0 | False |

`createDeviceHandle()` 矩阵（在 Ubuntu 22.04 x86_64 / Rosetta 容器内实测）：

| `loadBackendLib(socModel=)` | extension JSON `soc_model` | 结果 |
|---|---|---|
| 0 | 0 | SUCCESS |
| 0 | **87** | SUCCESS |
| 0 | 660 | FAILURE — `Unknown config socModel 0` |
| 87 | 87 | SUCCESS |
| 660 | 任意 | FAILURE — `Unsupported SnapdragonModel = 0` |

端到端 `qairt.compile()`（tiny ONNX → DLC → HTP context binary，纯 Python API）：

| `soc_details` | 结果 |
|---|---|
| `chipset:SM8850;dsp_arch:v81;soc_model:660` | `ContextBinaryGenerationError: (CREATE_DEVICE: 11, ...)` — 与线上报错完全一致 |
| `chipset:SM8850;dsp_arch:v81;soc_model:87` | **OK** |
| `chipset:SM8850` | **OK** |

x86_64 离线编译本身没有问题，Python API 也不缺接口。

## 已排除的方向

原 brief 的四条调查方向都不需要了：

1. `--htp_socs` 的 native 实现 — 无关。CLI 之所以能工作，是因为它接受
   *chipset 名字* (`SM8850`)，内部查表得到 87；我们绕过了同一张表。
2. `PythonBackendManager` 的隐藏路径 — 存在但非必需。
   `loadBackendLib(backendLibKey, backendLibPath, dlOpenLatency=0, socModel=0)`
   确实有一个未被 `native_executor.py` 使用的 `socModel` 参数，但上表证明：
   只要 extension JSON 里的值正确，`socModel=0` 也能成功。这条路是死胡同。
3. `LIBNATIVE_CONFIG` — 与本问题无关，那行输出是常规 info。
4. `qairt.compile()` 的 target 传递 — 没有遗漏。

## 修复

`soc_model` 全仓从 660 改为 87：

- `harness/constraints.json`、`qairt-agent.toml` — 受控 pin
- `src/qairt_agent/contracts.py` — `TargetSpec.soc_model: Literal[87] = 87`
- `examples/*.json`（含 `legacy/`）
- `tests/` 中的相关断言
- `src/qairt_agent/qairt_adapter/adapter.py` — 两处 `socModel 0` 诊断信息原本把
  问题误判为"已知 SDK x86_64 缺陷"，已改为指出枚举与 soc_id 的混淆

验证：`.venv/bin/pytest -q` → 493 passed, 2 skipped；
`qairt-agent doctor` → `target SM8850/v81/soc_model 87 (expected SM8850/v81/87)`。

**注意：** 这改变了 target 身份，已有 build 的复用会失效，需要重新编译。

## 防复发

`soc_model` 和 `soc_id` 是两个不同的命名空间，唯一权威来源分别是：

- `soc_model` → `include/QNN/QnnTypes.h` 的 `Qnn_SocModel_t`
- `soc_id` → `/sys/devices/soc0/soc_id`（仅用于在设备上识别芯片名）

新增芯片时，优先只写 `chipset:<NAME>` 让 SDK 查表；确实要显式写数字时，
以 `DeviceFactory.get_device_soc_details("HTP", "<CHIPSET>")` 的返回值为准，
不要用从设备上读到的 `soc_id`。
