"""Slice a Qwen3 ONNX model to keep only N decoder layers for debugging.

Usage:
    python scripts/slice_onnx_model.py --num-layers 2

Reads from: models/qwen3_4b/Qwen3-4B_onnx_w4a16/
Writes to:  models/Qwen3-4B_onnx_w4a16_{N}layer/
"""

import argparse
import json
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path

import onnx
from onnx import helper


def parse_args():
    parser = argparse.ArgumentParser(description="Slice ONNX model to N layers")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--src",
        type=str,
        default="models/qwen3_4b/Qwen3-4B_onnx_w4a16",
    )
    parser.add_argument("--dst", type=str, default=None)
    return parser.parse_args()


def get_layer_num(name):
    """Extract layer number from a name like /model/model/layers.15/... or model.model.layers.15.xxx"""
    m = re.search(r"layers\.(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def backward_bfs(output_tensors, tensor_producers, init_names, input_names):
    """Find all nodes needed to produce the given output tensors."""
    needed_nodes = set()
    queue = deque(output_tensors)
    visited_tensors = set()

    while queue:
        tensor = queue.popleft()
        if tensor in visited_tensors:
            continue
        visited_tensors.add(tensor)

        if tensor in init_names or tensor in input_names or not tensor:
            continue

        if tensor in tensor_producers:
            node = tensor_producers[tensor]
            if node.name not in needed_nodes:
                needed_nodes.add(node.name)
                for inp in node.input:
                    if inp:
                        queue.append(inp)

    return needed_nodes


def main():
    args = parse_args()
    num_layers = args.num_layers
    src_dir = Path(args.src)
    dst_dir = Path(args.dst) if args.dst else Path(f"models/Qwen3-4B_onnx_w4a16_{num_layers}layer")
    if num_layers <= 0:
        raise ValueError("--num-layers must be positive")

    with open(src_dir / "config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    total_layers = config.get("num_hidden_layers")
    if (
        isinstance(total_layers, bool)
        or not isinstance(total_layers, int)
        or total_layers <= 0
    ):
        raise ValueError(
            "config.json must contain a positive integer num_hidden_layers"
        )
    if num_layers > total_layers:
        raise ValueError(
            f"--num-layers={num_layers} exceeds source model layer count "
            f"{total_layers}"
        )
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) < num_layers:
        raise ValueError(
            "config.json layer_types must be a list covering every kept layer"
        )

    print(f"Slicing model to {num_layers} layers")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load model graph (no external data)
    print("\nLoading ONNX model graph...")
    model = onnx.load(str(src_dir / "model.onnx"), load_external_data=False)
    graph = model.graph

    init_names = set(i.name for i in graph.initializer)
    input_names = set(i.name for i in graph.input)
    # Step 1: Rewire norm's Cast node BEFORE BFS
    # The norm Cast node takes layer 35 output; change it to layer (N-1) output
    last_layer = num_layers - 1
    old_norm_input = f"/model/model/layers.{total_layers - 1}/Add_1_output_0"
    new_norm_input = f"/model/model/layers.{last_layer}/Add_1_output_0"

    print(f"\nStep 1: Rewiring norm input")
    print(f"  /model/model/norm/Cast: '{old_norm_input}' -> '{new_norm_input}'")

    rewired = False
    for node in graph.node:
        if node.name == "/model/model/norm/Cast":
            for i, inp in enumerate(node.input):
                if inp == old_norm_input:
                    node.input[i] = new_norm_input
                    rewired = True
                    break
            break
    if not rewired:
        raise RuntimeError(
            "could not rewire /model/model/norm/Cast from "
            f"{old_norm_input!r}; source graph layout is unsupported"
        )

    # Step 2: Build graph maps after rewiring
    tensor_producers = {}
    for node in graph.node:
        for out in node.output:
            if out:
                tensor_producers[out] = node

    # Step 3: Determine desired I/O
    desired_inputs = ["input_ids", "attention_mask", "position_ids_cos", "position_ids_sin"]
    desired_outputs = ["logits"]
    for i in range(num_layers):
        desired_inputs.extend([f"past_key_{i}_in", f"past_value_{i}_in"])
        desired_outputs.extend([f"past_key_{i}_out", f"past_value_{i}_out"])

    desired_input_set = set(desired_inputs)
    desired_output_set = set(desired_outputs)

    print(f"\nStep 2: Backward BFS from {len(desired_outputs)} outputs...")
    needed_nodes = backward_bfs(desired_outputs, tensor_producers, init_names, input_names)
    print(f"  Needed nodes: {len(needed_nodes)}")

    # Sanity check: no nodes from layers >= num_layers should be needed
    bad_layers = set()
    for name in needed_nodes:
        ln = get_layer_num(name)
        if ln is not None and ln >= num_layers:
            bad_layers.add(ln)
    if bad_layers:
        raise RuntimeError(
            "backward traversal reached removed layers "
            f"{sorted(bad_layers)}; refusing to publish an invalid slice"
        )
    else:
        print(f"  OK: no nodes from layers >= {num_layers}")

    # Step 4: Collect kept nodes (preserving original order for topological validity)
    print(f"\nStep 3: Building new graph...")
    kept_nodes = [node for node in graph.node if node.name in needed_nodes]
    print(f"  Kept {len(kept_nodes)} / {len(graph.node)} nodes")

    # Step 5: Filter initializers to only those consumed by kept nodes
    used_tensors = set()
    for node in kept_nodes:
        for inp in node.input:
            if inp:
                used_tensors.add(inp)

    kept_initializers = [init for init in graph.initializer if init.name in used_tensors]
    print(f"  Kept {len(kept_initializers)} / {len(graph.initializer)} initializers")

    # Step 6: Filter graph inputs/outputs
    new_inputs = [inp for inp in graph.input if inp.name in desired_input_set]
    new_outputs = [out for out in graph.output if out.name in desired_output_set]
    missing_inputs = sorted(
        desired_input_set - {item.name for item in new_inputs}
    )
    missing_outputs = sorted(
        desired_output_set - {item.name for item in new_outputs}
    )
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            "source graph does not expose the requested slice ABI: "
            f"missing_inputs={missing_inputs}, "
            f"missing_outputs={missing_outputs}"
        )
    print(f"  Inputs: {len(new_inputs)}, Outputs: {len(new_outputs)}")

    # Step 7: Keep relevant value_info
    produced = set()
    consumed = set()
    for node in kept_nodes:
        for out in node.output:
            if out:
                produced.add(out)
        for inp in node.input:
            if inp:
                consumed.add(inp)

    kept_value_info = [vi for vi in graph.value_info if vi.name in produced and vi.name in consumed]

    # Step 8: Assemble new model
    new_graph = helper.make_graph(
        kept_nodes,
        f"qwen3_{num_layers}layer",
        new_inputs,
        new_outputs,
        initializer=kept_initializers,
        value_info=kept_value_info,
    )

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    new_model.producer_name = model.producer_name
    new_model.producer_version = model.producer_version

    # Validate
    print("\nStep 4: Validating...")
    onnx.checker.check_model(new_model, full_check=False)
    print("  Validation passed")

    # Step 9: Save model.onnx and extract relevant data from original model.data
    print(f"\nStep 5: Saving to {dst_dir}...")
    save_with_external_data(new_model, src_dir, dst_dir)
    print("  Saved model.onnx + model.data")

    # Step 10: Filter encodings
    print("\nStep 6: Filtering encodings...")
    filter_encodings(src_dir / "model.encodings", dst_dir / "model.encodings", num_layers)

    # Step 11: Update config.json
    print("\nStep 7: Updating config.json...")
    config["num_hidden_layers"] = num_layers
    config["layer_types"] = layer_types[:num_layers]
    config["max_window_layers"] = num_layers

    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  num_hidden_layers = {num_layers}")

    # Step 12: Copy tokenizer files
    print("\nStep 8: Copying tokenizer files...")
    for fname in [
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "tool_versions.yaml",
    ]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
            print(f"  {fname}")

    print(f"\nDone!")
    print(f"  Nodes: {len(graph.node)} -> {len(kept_nodes)}")
    print(f"  Initializers: {len(graph.initializer)} -> {len(kept_initializers)}")
    print(f"  Output: {dst_dir}")


def save_with_external_data(new_model, src_dir, dst_dir):
    """Save model.onnx and create a new model.data with only the kept tensors.

    Since we loaded the original model without external data, the initializer
    tensors still carry their original (offset, length) metadata pointing into
    the source model.data.  We copy only those byte ranges into a new contiguous
    model.data and rewrite the offsets accordingly.
    """
    src_data_path = src_dir / "model.data"
    dst_data_path = dst_dir / "model.data"

    # Work on the actual initializers inside the new model
    graph_inits = list(new_model.graph.initializer)

    # Collect (init, src_offset, length) for tensors stored externally
    chunks = []
    for init in graph_inits:
        ed_map = {ed.key: ed.value for ed in init.external_data}
        if "offset" in ed_map and "length" in ed_map:
            chunks.append((init, int(ed_map["offset"]), int(ed_map["length"])))

    # Sort by source offset for sequential reads
    chunks.sort(key=lambda x: x[1])

    total_bytes = sum(length for _, _, length in chunks)
    print(f"  Extracting {len(chunks)} tensors ({total_bytes / 1e9:.2f} GB) from {src_data_path}")

    # Write new model.data and update offsets
    new_offset = 0
    with open(src_data_path, "rb") as src_f, open(dst_data_path, "wb") as dst_f:
        for init, src_offset, length in chunks:
            src_f.seek(src_offset)
            remaining = length
            while remaining > 0:
                buf_size = min(remaining, 64 * 1024 * 1024)  # 64 MB chunks
                data = src_f.read(buf_size)
                if not data:
                    raise IOError(f"Unexpected EOF in {src_data_path}")
                dst_f.write(data)
                remaining -= len(data)

            # Update external_data metadata to new offset
            for ed in init.external_data:
                if ed.key == "offset":
                    ed.value = str(new_offset)
                elif ed.key == "location":
                    ed.value = "model.data"
            new_offset += length

    print(f"  Wrote {new_offset / 1e9:.2f} GB to {dst_data_path}")

    # Save model.onnx (graph references model.data via external_data fields)
    onnx.save_model(new_model, str(dst_dir / "model.onnx"))


def filter_encodings(src_path, dst_path, num_layers):
    """Filter encodings JSON to keep only entries for kept layers + shared components."""
    with open(src_path) as f:
        encodings = json.load(f)

    for section in ("activation_encodings", "param_encodings"):
        if section not in encodings:
            continue
        entries = encodings[section]
        if not isinstance(entries, list):
            raise ValueError(
                f"{src_path} field {section!r} must be a list; refusing to "
                "copy unfiltered encodings"
            )
        orig_count = len(entries)
        filtered = [
            entry for entry in entries
            if (layer_num := get_layer_num(entry.get("name", ""))) is None
            or layer_num < num_layers
        ]
        encodings[section] = filtered
        print(f"  {section}: {orig_count} -> {len(filtered)}")

    with open(dst_path, "w") as f:
        json.dump(encodings, f)
    print(f"  Saved: {dst_path}")


if __name__ == "__main__":
    main()
