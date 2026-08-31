from __future__ import annotations

import unittest

from qairt_agent.diagnostics.device_metrics import (
    DEVICE_EXECUTION_METER,
    DEVICE_EXECUTION_SCHEMA,
    DeviceMetricsError,
    aggregate_device_executions,
    parse_device_execution,
)


# Captured verbatim from QAIRT 2.49.0.260730 on the SM8750 handset (serial
# RFCY30B296K) running the tiny acceptance graph, via
# qairt.Profiler(context={"level": "detailed"}).  The wall-clock sample around
# the same call measured ~4900 ms.
REAL_REPORT: dict[str, object] = {
    "header": {"artifact_type": "QNN_PROFILE"},
    "metadata": {
        "appName": "qnn-net-run",
        "appVersion": "v2.49.0.260730134355",
        "backendVersion": "v2.49.0.260730134355",
        "qnnProfileViewerVersion": "v2.49.0.260730134355",
    },
    "messages": [
        {
            "method": "BACKEND_CREATE_FROM_BINARY",
            "profilingEvents": [
                {"identifier": "RPC (load binary) time", "unit": "MICROSEC", "type": "", "value": 3104},
                {"identifier": "QNN accelerator (load binary) time", "unit": "MICROSEC", "type": "", "value": 2701},
                {"identifier": "Accelerator (load binary) time", "unit": "MICROSEC", "type": "", "value": 2258},
                {"identifier": "QNN (load binary) time", "unit": "MICROSEC", "type": "INIT", "value": 9009},
            ],
        },
        {
            "method": "BACKEND_EXECUTE",
            "profilingEvents": [
                {"identifier": "Number of HVX threads used", "unit": "COUNT", "type": "", "value": 6},
                {"identifier": "RPC (execute) time", "unit": "MICROSEC", "type": "", "value": 2886},
                {"identifier": "QNN accelerator (execute) time", "unit": "MICROSEC", "type": "", "value": 2381},
                {"identifier": "Num times yield occured", "unit": "COUNT", "type": "", "value": 0},
                {"identifier": "Time for initial VTCM acquire", "unit": "MICROSEC", "type": "", "value": 804},
                {
                    "identifier": "Time for HVX + HMX power on and acquire",
                    "unit": "MICROSEC",
                    "type": "",
                    "value": 29218,
                },
                {
                    "identifier": "Accelerator (execute) time (cycles)",
                    "unit": "CYCLES",
                    "type": "",
                    "value": 29239,
                    "sub-events": [
                        {"identifier": "Input OpId_2 (cycles)", "unit": "CYCLES", "type": "NODE", "value": 8405},
                        {"identifier": "fc:OpId_17 (cycles)", "unit": "CYCLES", "type": "NODE", "value": 6137},
                        {"identifier": "bias_add:OpId_21 (cycles)", "unit": "CYCLES", "type": "NODE", "value": 0},
                        {"identifier": "act:OpId_23 (cycles)", "unit": "CYCLES", "type": "NODE", "value": 8567},
                        {"identifier": "Output OpId_3 (cycles)", "unit": "CYCLES", "type": "NODE", "value": 6130},
                    ],
                },
                {"identifier": "Accelerator (execute) time", "unit": "MICROSEC", "type": "", "value": 1734},
                {
                    "identifier": "Accelerator (execute excluding wait) time",
                    "unit": "MICROSEC",
                    "type": "",
                    "value": 77,
                },
                {"identifier": "QNN (execute) time", "unit": "MICROSEC", "type": "EXECUTE", "value": 3001},
            ],
        },
        {
            "method": "APP_EXECUTE_IPS",
            "profilingEvents": [
                {"identifier": "numInferences", "unit": "COUNT", "type": "EXECUTE", "value": 1},
                {"identifier": "duration", "unit": "MICROSEC", "type": "EXECUTE", "value": 3309},
            ],
        },
        {
            "method": "BACKEND_DEINIT",
            "profilingEvents": [
                {"identifier": "RPC (deinit) time", "unit": "MICROSEC", "type": "", "value": 1794},
                {"identifier": "QNN (deinit) time", "unit": "MICROSEC", "type": "DEINIT", "value": 19359},
            ],
        },
    ],
}


class DeviceExecutionParsingTests(unittest.TestCase):
    def test_real_report_yields_the_device_side_numbers(self) -> None:
        block = parse_device_execution(REAL_REPORT)

        self.assertEqual(block["schema"], DEVICE_EXECUTION_SCHEMA)
        self.assertEqual(block["policy"], "report_only")
        # The number the NPU actually spent, and the two wider device figures.
        self.assertEqual(block["accelerator_compute_us"], 77.0)
        self.assertEqual(block["accelerator_execute_us"], 1734.0)
        self.assertEqual(block["qnn_execute_us"], 3001.0)
        self.assertEqual(block["accelerator_execute_cycles"], 29239.0)
        self.assertEqual(block["producer"]["app_name"], "qnn-net-run")

    def test_per_op_cycles_are_carried_through_including_zero(self) -> None:
        block = parse_device_execution(REAL_REPORT)
        cycles = {item["identifier"]: item["cycles"] for item in block["per_op_cycles"]}

        self.assertEqual(
            cycles,
            {
                "Input OpId_2 (cycles)": 8405.0,
                "fc:OpId_17 (cycles)": 6137.0,
                "bias_add:OpId_21 (cycles)": 0.0,
                "act:OpId_23 (cycles)": 8567.0,
                "Output OpId_3 (cycles)": 6130.0,
            },
        )
        # A zero-cycle operator is a real measurement, not a missing one.
        self.assertIn("bias_add:OpId_21 (cycles)", cycles)

    def test_per_process_overhead_is_reported_separately_from_execute(self) -> None:
        # Load-binary and deinit are per-qnn-net-run-process costs.  They are
        # inside every wall-clock sample but are not execute time, so they must
        # not be folded into the execute numbers.
        block = parse_device_execution(REAL_REPORT)
        overhead = block["per_process_overhead_us"]

        self.assertEqual(overhead["QNN (load binary) time"], 9009.0)
        self.assertEqual(overhead["QNN (deinit) time"], 19359.0)
        self.assertNotIn("QNN (load binary) time", block["execute_events_us"])
        # Power-on is charged to the execute message by QAIRT itself.
        self.assertEqual(
            block["execute_events_us"]["Time for HVX + HMX power on and acquire"],
            29218.0,
        )

    def test_the_block_states_it_is_not_the_timed_samples(self) -> None:
        block = parse_device_execution(REAL_REPORT)
        self.assertEqual(
            block["claim_scope"], "one_profiled_execute_not_the_timed_samples"
        )
        self.assertEqual(block["sample_unit"], "one_profiled_graph_execute")
        self.assertIsNone(block["profiler_option"])

    def test_non_microsecond_events_are_not_treated_as_times(self) -> None:
        block = parse_device_execution(REAL_REPORT)
        # COUNT events would otherwise land in the time map as bare numbers.
        self.assertNotIn("Number of HVX threads used", block["execute_events_us"])
        self.assertNotIn("Num times yield occured", block["execute_events_us"])


def _report_with(compute_us: int, execute_us: int, fc_cycles: int) -> dict[str, object]:
    return {
        "metadata": {"appName": "qnn-net-run"},
        "messages": [
            {
                "method": "BACKEND_EXECUTE",
                "profilingEvents": [
                    {
                        "identifier": "Accelerator (execute excluding wait) time",
                        "unit": "MICROSEC",
                        "value": compute_us,
                    },
                    {
                        "identifier": "Accelerator (execute) time",
                        "unit": "MICROSEC",
                        "value": execute_us,
                    },
                    {
                        "identifier": "Accelerator (execute) time (cycles)",
                        "unit": "CYCLES",
                        "value": fc_cycles * 2,
                        "sub-events": [
                            {
                                "identifier": "fc:OpId_17 (cycles)",
                                "unit": "CYCLES",
                                "type": "NODE",
                                "value": fc_cycles,
                            }
                        ],
                    },
                ],
            }
        ],
    }


class DeviceExecutionAggregationTests(unittest.TestCase):
    def test_headline_scalars_are_the_mean_of_the_samples(self) -> None:
        blocks = [
            parse_device_execution(_report_with(compute, compute * 10, compute * 100))
            for compute in (70, 80, 90)
        ]

        aggregated = aggregate_device_executions(blocks)

        self.assertEqual(aggregated["statistic"], "mean")
        self.assertEqual(aggregated["sample_count"], 3)
        self.assertEqual(aggregated["accelerator_compute_us"], 80.0)
        self.assertEqual(aggregated["accelerator_execute_us"], 800.0)
        # Per-op cycles are averaged too, not taken from one arbitrary sample.
        self.assertEqual(
            aggregated["per_op_cycles"],
            [{"identifier": "fc:OpId_17 (cycles)", "cycles": 8000.0}],
        )

    def test_spread_and_raw_samples_are_kept(self) -> None:
        # An average of ten hides a bimodal device; the reader must be able to
        # see that without rerunning the benchmark.
        blocks = [
            parse_device_execution(_report_with(compute, compute * 10, compute * 100))
            for compute in (70, 80, 90)
        ]

        aggregated = aggregate_device_executions(blocks)
        spread = aggregated["spread"]["accelerator_compute_us"]

        self.assertEqual(aggregated["samples"]["accelerator_compute_us"], [70.0, 80.0, 90.0])
        self.assertEqual(spread["p50"], 80.0)
        self.assertEqual(spread["min"], 70.0)
        self.assertEqual(spread["max"], 90.0)
        self.assertGreater(spread["stddev"], 0.0)

    def test_single_sample_reports_zero_spread_not_an_error(self) -> None:
        aggregated = aggregate_device_executions(
            [parse_device_execution(_report_with(77, 1734, 6137))]
        )
        self.assertEqual(aggregated["sample_count"], 1)
        self.assertEqual(aggregated["spread"]["accelerator_compute_us"]["stddev"], 0.0)

    def test_the_block_names_its_meter_and_lane(self) -> None:
        # The GenAI lane cannot use this meter, so the block says which one it
        # is rather than leaving the two lanes' numbers to look interchangeable.
        aggregated = aggregate_device_executions(
            [parse_device_execution(_report_with(77, 1734, 6137))]
        )
        self.assertEqual(aggregated["meter"], DEVICE_EXECUTION_METER)
        self.assertEqual(aggregated["lane"], "low_level")
        self.assertEqual(
            aggregated["claim_scope"], "profiled_executes_not_the_host_wall_samples"
        )

    def test_production_latency_is_accelerator_compute_with_its_spread(self) -> None:
        # Maintainer decision: production latency is the accelerator's own
        # compute time. Its small absolute value makes it the most dispersed
        # metric in the block, so the report must never present it without its
        # dispersion.
        blocks = [
            parse_device_execution(_report_with(compute, compute * 10, compute * 100))
            for compute in (70, 80, 90)
        ]

        aggregated = aggregate_device_executions(blocks)

        self.assertEqual(aggregated["production_latency_source"], "accelerator_compute_us")
        self.assertEqual(aggregated["production_latency_us"], 80.0)
        self.assertEqual(
            aggregated["production_latency_us"], aggregated["accelerator_compute_us"]
        )
        self.assertAlmostEqual(
            aggregated["production_latency_cv_percent"],
            100.0 * aggregated["spread"]["accelerator_compute_us"]["stddev"] / 80.0,
        )
        self.assertIn("excluding wait", aggregated["production_latency_note"])
        self.assertIn(
            "production_latency_cv_percent", aggregated["production_latency_note"]
        )

    def test_production_latency_is_absent_when_the_backend_did_not_report_it(
        self,
    ) -> None:
        # "Accelerator (execute excluding wait) time" is optional in the log.
        # Falling back to a wider metric under the production-latency name
        # would silently change what the number means.
        report = {
            "messages": [
                {
                    "method": "BACKEND_EXECUTE",
                    "profilingEvents": [
                        {
                            "identifier": "Accelerator (execute) time",
                            "unit": "MICROSEC",
                            "value": 1734,
                        }
                    ],
                }
            ]
        }
        aggregated = aggregate_device_executions([parse_device_execution(report)])

        self.assertNotIn("production_latency_us", aggregated)
        self.assertEqual(aggregated["accelerator_execute_us"], 1734.0)

    def test_aggregating_nothing_fails_closed(self) -> None:
        with self.assertRaises(DeviceMetricsError):
            aggregate_device_executions([])


class DeviceExecutionFailClosedTests(unittest.TestCase):
    def test_report_without_an_execute_message_fails_closed(self) -> None:
        report = {"messages": [message for message in REAL_REPORT["messages"]  # type: ignore[index]
                               if message["method"] != "BACKEND_EXECUTE"]}
        with self.assertRaises(DeviceMetricsError) as caught:
            parse_device_execution(report)
        self.assertIn("BACKEND_EXECUTE", str(caught.exception))

    def test_execute_message_without_accelerator_time_fails_closed(self) -> None:
        report = {
            "messages": [
                {
                    "method": "BACKEND_EXECUTE",
                    "profilingEvents": [
                        {
                            "identifier": "Number of HVX threads used",
                            "unit": "COUNT",
                            "value": 6,
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(DeviceMetricsError):
            parse_device_execution(report)

    def test_a_report_that_is_not_a_profiling_report_fails_closed(self) -> None:
        with self.assertRaises(DeviceMetricsError):
            parse_device_execution({"header": {}})
        with self.assertRaises(DeviceMetricsError):
            parse_device_execution(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
