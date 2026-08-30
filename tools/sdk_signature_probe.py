"""Import-based signature probe for every QAIRT surface the adapter binds.

Run this **inside the worker container**, where the SDK actually imports; on a
macOS/arm64 host its native extensions do not load and a probe would report
false absences. Exits non-zero and names anything missing, so an SDK upgrade
cannot land on the assumption that a bound surface still exists.

    container run --rm --platform linux/amd64 --rosetta --no-dns \
      --workdir /opt/qairt-agent \
      --env QAIRT_SDK_ROOT=/opt/qairt --env QNN_SDK_ROOT=/opt/qairt \
      --env PYTHONPATH=/opt/qairt-agent/src:/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN \
      --env LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang \
      --mount type=bind,source=<sdk>,target=/opt/qairt,readonly \
      --mount type=bind,source=<repo>/tools,target=/probe,readonly \
      <worker-image> /opt/venv/bin/python /probe/sdk_signature_probe.py
"""
import importlib
import inspect
import json

PROBES = [
    ("qairt", "convert"), ("qairt", "compile"), ("qairt", "load"),
    ("qairt", "CompileConfig"), ("qairt", "CalibrationConfig"),
    ("qairt", "Device"), ("qairt", "RemoteDeviceIdentifier"),
    ("qairt", "DevicePlatformType"), ("qairt", "Profiler"),
    ("qairt.api.transforms._transform", "transform"),
    ("qairt.api.transforms.model_transformer_config", "MhaConfig"),
    ("qairt.api.transforms.model_transformer_config", "SplitModelConfig"),
    ("qairt.api.configs.common", "DspArchitecture"),
    ("qairt.optimizer.onnx", "split_llm"),
    ("qairt.optimizer.onnx", "M2sStartPoint"),
    ("qairt.gen_ai_api.gen_ai_builder_factory", "GenAIBuilderFactory"),
    ("qairt.gen_ai_api.builders.qwen.builder", "Qwen3_5BuilderHTP"),
    ("qairt.gen_ai_api.builders.workflow_builder", "WorkflowBuilder"),
    ("qairt.gen_ai_api.builders.vision_encoder_builder_htp", "VisionEncoderBuilderHTP"),
    ("qairt.gen_ai_api.configs.workflow", "WorkflowGraph"),
    ("qairt.gen_ai_api.configs.workflow", "WorkflowNode"),
    ("qairt.gen_ai_api.configs.workflow", "WorkflowNodeRole"),
    ("qairt.gen_ai_api.containers.container_factory", "load_container"),
    ("qti.aisw.tools.core.modules.converter.quantizer_module", "QuantizerInputConfig"),
    ("qti.aisw.tools.core.modules.converter.quantizer_module", "QAIRTQuantizer"),
]
METHODS = [
    ("qairt.gen_ai_api.builders.qwen.builder", "Qwen3_5BuilderHTP",
     ("from_pretrained", "attach_model_for_arn", "build")),
    ("qairt", "CompileConfig", ("set_mode",)),
]

rows, missing = [], []
for module_name, attribute in PROBES:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        missing.append(f"{module_name}: IMPORT FAILED ({type(error).__name__}: {error})")
        continue
    target = getattr(module, attribute, None)
    if target is None:
        missing.append(f"{module_name}.{attribute}: MISSING")
        continue
    try:
        signature = str(inspect.signature(target))
    except (TypeError, ValueError):
        signature = "<no signature>"
    rows.append({"module": module_name, "attribute": attribute, "signature": signature})

for module_name, class_name, methods in METHODS:
    try:
        owner = getattr(importlib.import_module(module_name), class_name)
    except Exception as error:
        missing.append(f"{module_name}.{class_name}: {error}")
        continue
    for method in methods:
        function = getattr(owner, method, None)
        if function is None:
            missing.append(f"{class_name}.{method}: MISSING")
            continue
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = "<no signature>"
        rows.append({"module": f"{module_name}.{class_name}", "attribute": method,
                     "signature": signature})

print(json.dumps({"probed": len(rows) + len(missing), "present": len(rows),
                  "missing": missing, "rows": rows}, indent=1))
raise SystemExit(1 if missing else 0)
