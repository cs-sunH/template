#!/usr/bin/env python3
"""CLI availability and actual native response regression, without timing math."""
import argparse
import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

arguments = argparse.ArgumentParser()
arguments.add_argument('--library', required=True)
args, remaining = arguments.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
source = pathlib.Path(__file__).resolve().parents[1]
script = source / 'tools/estimate.py'
module_spec = importlib.util.spec_from_file_location('coarse_die_estimate_cli', script)
cli = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cli)


class ResultAvailabilityTests(unittest.TestCase):
    def test_mixed_rows_preserve_order_body_and_native_errors(self):
        good = {'status': 'ok_assumed', 'estimated_latency_ns': 17,
                'calibration': {'heldout_status': 'not_available'}}
        body = {'status': 'not_available', 'estimated_latency_ns': 42}
        native = {'status_code': 1, 'status': 'invalid_argument', 'reason': 'bad task'}
        raw = {'results': [good, body, None], 'errors': [None, None, native]}
        before = copy.deepcopy(raw)
        result, failed = cli.render_result(raw, 3)
        self.assertTrue(failed)
        self.assertEqual(result['status'], 'partial_error')
        self.assertEqual([x['index'] for x in result['items']], [0, 1, 2])
        self.assertEqual(result['items'][0]['result'], good)
        self.assertIsNone(result['items'][0]['error'])
        unavailable = result['items'][1]
        self.assertEqual(unavailable['status'], 'not_available')
        self.assertEqual(unavailable['result'], body)
        self.assertEqual(unavailable['status_code'], 0)  # successful ABI call
        self.assertEqual(unavailable['error']['reason'], 'prediction_not_available')
        self.assertEqual(result['items'][2]['error'], native)
        self.assertEqual(raw, before)

    def test_only_unavailable_is_not_global_success(self):
        raw = {'results': [{'status': 'not_available', 'estimated_latency_ns': 42}], 'errors': [None]}
        result, failed = cli.render_result(raw, 1)
        self.assertTrue(failed)
        self.assertEqual(result['status'], 'partial_error')
        self.assertEqual(result['items'][0]['result'], raw['results'][0])

    def test_missing_or_malformed_status_is_not_success(self):
        for value in [None, False, 7, {}, []]:
            with self.subTest(status=value):
                row = {'status': value, 'estimated_latency_ns': 42}
                for candidate in [row, {'estimated_latency_ns': 42}]:
                    result, failed = cli.render_result({'results': [candidate], 'errors': [None]}, 1)
                    self.assertTrue(failed)
                    self.assertEqual(result['items'][0]['status'], 'error')
                    self.assertEqual(result['items'][0]['error']['reason'], 'invalid_prediction_status')
                    self.assertEqual(result['items'][0]['result'], candidate)

    def test_nested_missing_calibration_does_not_reject_prediction(self):
        row = {'status': 'ok_assumed', 'estimated_latency_ns': 17,
               'heldout': {'status': 'not_available'}}
        result, failed = cli.render_result({'results': [row], 'errors': [None]}, 1)
        self.assertFalse(failed)
        self.assertEqual(result['status'], 'ok_assumed')
        self.assertIsNone(result['items'][0]['error'])


class NativeCliTests(unittest.TestCase):
    def call_cli(self, tasks):
        with tempfile.TemporaryDirectory(prefix='coarse_die_cli_') as directory:
            task_file = pathlib.Path(directory) / 'tasks.json'
            task_file.write_text(json.dumps(tasks))
            command = [sys.executable, str(script), '--library', args.library,
                       '--hardware', str(source / 'configs/example_hardware_explicit.json'),
                       '--tasks', str(task_file), '--output', '-']
            result = subprocess.run(command, text=True, capture_output=True)
            return result, json.loads(result.stdout)

    @staticmethod
    def task(layers):
        return {'op_kind': 'residual', 'tensor_shapes': {'m': 1, 'hidden': 257},
                'storage_dtype': 'bf16', 'layer_instances': layers, 'weight_passes': 1}

    def test_real_valid_r1_still_exits_zero(self):
        process, report = self.call_cli(self.task(1))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report['status'], 'ok_assumed')
        self.assertEqual(report['items'][0]['status'], 'ok_assumed')
        self.assertGreater(report['items'][0]['result']['estimated_latency_ns'], 0)

    def test_real_r3_body_exits_nonzero_without_losing_diagnostics(self):
        tasks = [self.task(1), self.task(3), {'op_kind': 'invalid_operation_for_cli_test'}]
        process, report = self.call_cli(tasks)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(report['status'], 'partial_error')
        good, unavailable, bad = report['items']
        self.assertIsNone(good['error'])
        self.assertEqual(unavailable['status'], 'not_available')
        self.assertEqual(unavailable['result']['status'], 'not_available')
        self.assertGreater(unavailable['result']['estimated_latency_ns'], 0)
        self.assertEqual(unavailable['error']['reason'], 'prediction_not_available')
        self.assertIsNone(bad['result'])
        self.assertIsNotNone(bad['error'])

    def test_real_r3_without_native_error_also_exits_nonzero(self):
        process, report = self.call_cli(self.task(3))
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(report['status'], 'partial_error')
        self.assertEqual(report['items'][0]['status_code'], 0)
        self.assertEqual(report['items'][0]['result']['status'], 'not_available')


if __name__ == '__main__':
    unittest.main()
