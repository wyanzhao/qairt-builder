from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from qairt_agent.families import (
    AutoArPolicy,
    FamilyConfigGenerator,
    FamilyId,
    OnnxInspector,
    QairtFactorySupport,
    build_split_plan,
    get_family_profile,
    resolve_family_profile,
    validate_weight_sharing_sources,
)


class FamilyProfileTests(unittest.TestCase):
    def test_all_qwen_profiles_resolve(self) -> None:
        dense = get_family_profile("qwen3")
        moe = get_family_profile("qwen3_moe")
        vision = get_family_profile("qwen3-vl")
        hybrid = get_family_profile("qwen3.5")
        thinker = get_family_profile("qwen3.5-omni-thinker")

        self.assertEqual(dense.factory_support, QairtFactorySupport.GENERIC_FALLBACK)
        self.assertTrue(moe.is_moe)
        self.assertTrue(vision.is_multimodal)
        self.assertEqual(hybrid.auto_ar_policy, AutoArPolicy.EXPERIMENTAL_FAIL_CLOSED)
        self.assertTrue(hybrid.is_hybrid_attention)
        self.assertIs(thinker, hybrid)

    def test_qwen35_start_points_match_sdk_contract(self) -> None:
        profile = get_family_profile(FamilyId.QWEN3_5)
        self.assertEqual(len(profile.mha_start_points), 4)
        self.assertEqual(profile.mha_start_points[0].axis, 1)
        self.assertEqual(profile.mha_start_points[1].axis, 2)
        self.assertEqual(profile.mha_start_points[1].split_map, {4096: 256})
        self.assertIn("recurrent_state", profile.mha_start_points[2].output_name_regex)
        self.assertIn("conv_state", profile.mha_start_points[3].output_name_regex)

    def test_qwen35_single_source_is_not_preemptively_rejected(self) -> None:
        # The adapter's validation evidence gates fail closed later.  A derived
        # single-base-model flow itself is a supported request.
        validate_weight_sharing_sources(FamilyId.QWEN3_5, ["derived", "derived"])
        with self.assertRaises(ValueError):
            validate_weight_sharing_sources(FamilyId.QWEN3_5, ["unknown"])

    def test_resolve_architectures(self) -> None:
        self.assertEqual(
            resolve_family_profile({"architectures": ["Qwen3MoeForCausalLM"]}).family,
            FamilyId.QWEN3_MOE,
        )
        self.assertEqual(
            resolve_family_profile(
                {
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                    "text_config": {"model_type": "qwen3"},
                }
            ).family,
            FamilyId.QWEN3_VL,
        )
        self.assertEqual(
            resolve_family_profile(
                {"architectures": ["Qwen3_5OmniThinkerForConditionalGeneration"]}
            ).family,
            FamilyId.QWEN3_5,
        )


class SplitPlanTests(unittest.TestCase):
    def test_balanced_decoder_ranges_and_edge_slices(self) -> None:
        plan = build_split_plan(
            7,
            decoder_slices=3,
            split_embedding=True,
            split_lm_head=True,
        )
        self.assertEqual(plan.num_splits, 5)
        # split_llm folds layer 6 into the lm_head split, so only 6 layers are
        # distributed across the three decoder slices.
        self.assertEqual(
            [(item.layer_start, item.layer_end) for item in plan.decoder_slices],
            [(0, 2), (2, 4), (4, 6)],
        )
        self.assertEqual(plan.folded_lm_head_layer, 6)
        self.assertEqual(
            plan.to_qairt_kwargs(),
            {"num_splits": 5, "split_embedding": True, "split_lm_head": True},
        )

    def test_lm_head_split_folds_the_final_decoder_layer(self) -> None:
        # QAIRT 2.49 llm_splitter pops the last post-FFN residual add before it
        # distributes boundaries: 28 layers over 4 decoder splits becomes
        # 7/7/7/6 with layer 27 folded into the lm_head split.
        plan = build_split_plan(
            28,
            decoder_slices=4,
            split_embedding=True,
            split_lm_head=True,
        )
        self.assertEqual(
            [item.layer_count for item in plan.decoder_slices],
            [7, 7, 7, 6],
        )
        self.assertEqual(
            [(item.layer_start, item.layer_end) for item in plan.decoder_slices],
            [(0, 7), (7, 14), (14, 21), (21, 27)],
        )
        self.assertEqual(plan.distributed_decoder_layers, 27)
        self.assertEqual(plan.folded_lm_head_layer, 27)
        self.assertEqual(plan.num_splits, 6)

    def test_without_lm_head_split_every_layer_stays_in_a_decoder_slice(
        self,
    ) -> None:
        plan = build_split_plan(
            28,
            decoder_slices=4,
            split_embedding=True,
            split_lm_head=False,
        )
        self.assertEqual(
            [item.layer_count for item in plan.decoder_slices],
            [7, 7, 7, 7],
        )
        self.assertEqual(plan.distributed_decoder_layers, 28)
        self.assertIsNone(plan.folded_lm_head_layer)

    def test_reject_more_decoder_slices_than_layers(self) -> None:
        with self.assertRaises(ValueError):
            build_split_plan(2, decoder_slices=3)

    def test_reject_decoder_slices_beyond_the_distributed_layers(self) -> None:
        # 4 layers with a folded lm_head leaves only 3 distributable layers.
        build_split_plan(4, decoder_slices=3, split_lm_head=True)
        with self.assertRaises(ValueError) as caught:
            build_split_plan(4, decoder_slices=4, split_lm_head=True)
        self.assertIn("split_llm distributes", str(caught.exception))


class FamilyConfigTests(unittest.TestCase):
    def test_dense_config_generation(self) -> None:
        generated = FamilyConfigGenerator().generate(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
            },
            decoder_slices=4,
        )
        self.assertEqual(generated.family, FamilyId.QWEN3_DENSE)
        self.assertEqual(generated.head_dim, 128)
        self.assertEqual(generated.native_kv_head_shape, (8, 128))
        self.assertEqual(len(generated.split_plan.decoder_slices), 4)

    def test_vl_uses_nested_text_config(self) -> None:
        generated = FamilyConfigGenerator().generate(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "max_position_embeddings": 8192,
                },
            }
        )
        self.assertEqual(generated.family, FamilyId.QWEN3_VL)
        self.assertEqual(generated.source_config_container, "text_config")
        self.assertEqual(generated.num_hidden_layers, 24)

    def test_qwen35_auto_ar_is_warned_not_rejected(self) -> None:
        generated = FamilyConfigGenerator().generate(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "hidden_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
            },
            ar_values=(1, 128),
        )
        self.assertTrue(generated.warnings)
        self.assertEqual(generated.auto_ar_policy, AutoArPolicy.EXPERIMENTAL_FAIL_CLOSED)


class LazyInspectorTests(unittest.TestCase):
    @staticmethod
    def _value_info(name: str, dimensions: tuple[int | str | None, ...], elem_type: int = 1):
        dims = []
        for value in dimensions:
            if isinstance(value, str):
                dims.append(
                    SimpleNamespace(
                        dim_param=value,
                        dim_value=0,
                        HasField=lambda _field, result=False: result,
                    )
                )
            elif value is None:
                dims.append(
                    SimpleNamespace(
                        dim_param="",
                        dim_value=0,
                        HasField=lambda _field, result=False: result,
                    )
                )
            else:
                dims.append(
                    SimpleNamespace(
                        dim_param="",
                        dim_value=value,
                        HasField=lambda _field, result=True: result,
                    )
                )
        return SimpleNamespace(
            name=name,
            type=SimpleNamespace(
                tensor_type=SimpleNamespace(
                    elem_type=elem_type,
                    shape=SimpleNamespace(dim=dims),
                )
            ),
        )

    def test_onnx_is_loaded_only_when_inspection_runs(self) -> None:
        loaded: list[str] = []
        runtime_input = self._value_info("input_ids", (1, "seq"))
        weight_input = self._value_info("weight", (8, 8))
        output = self._value_info("logits", (1, "seq", 8))
        model = SimpleNamespace(
            graph=SimpleNamespace(
                name="decoder",
                input=[runtime_input, weight_input],
                output=[output],
                initializer=[SimpleNamespace(name="weight")],
                node=[
                    SimpleNamespace(
                        name="matmul",
                        op_type="MatMul",
                        input=["input_ids", "weight"],
                        output=["logits"],
                        domain="",
                    )
                ],
            ),
            metadata_props=[SimpleNamespace(key="family", value="qwen3")],
        )

        class DataType:
            @staticmethod
            def Name(value: int) -> str:
                return {1: "FLOAT"}[value]

        fake_onnx = SimpleNamespace(
            load=lambda path, load_external_data: model,
            TensorProto=SimpleNamespace(DataType=DataType),
        )

        def loader(name: str):
            loaded.append(name)
            return fake_onnx

        inspector = OnnxInspector(module_loader=loader)
        self.assertEqual(loaded, [])
        info = inspector.inspect(Path("/not/loaded/model.onnx"))
        self.assertEqual(loaded, ["onnx"])
        self.assertEqual([item.name for item in info.inputs], ["input_ids"])
        self.assertEqual(info.inputs[0].shape, (1, "seq"))
        self.assertEqual(info.nodes[0].op_type, "MatMul")

    def test_external_data_paths_are_resolved_without_loading_weights(self) -> None:
        model = SimpleNamespace(
            graph=SimpleNamespace(
                initializer=[
                    SimpleNamespace(
                        external_data=[
                            SimpleNamespace(key="location", value="weights/model.data")
                        ]
                    )
                ]
            )
        )
        fake_onnx = SimpleNamespace(
            load=lambda path, load_external_data: model,
        )
        inspector = OnnxInspector(module_loader=lambda _name: fake_onnx)
        paths = inspector.external_data_paths("/models/model.onnx")
        self.assertEqual(paths, (Path("/models/weights/model.data"),))


if __name__ == "__main__":
    unittest.main()
