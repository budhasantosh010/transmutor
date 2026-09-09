from __future__ import annotations

import unittest

from experiments.v837_primitive_invention.v837al.authorization import freeze_gate, assert_authorized
from experiments.v837_primitive_invention.v837al.source_integrity import run_source_integrity
from experiments.v837_primitive_invention.v837al.pairing import freeze_pairs
from experiments.v837_primitive_invention.v837al.legacy_baseline_reproduction import reproduce_baselines


class TestV837alBootstrap(unittest.TestCase):
    def test_00_freeze_gate_before_adapter_results(self):
        gate=freeze_gate()
        self.assertEqual(gate['starting_sha'],'802478151a5ab0d4cbbe099e3c18f056e8ce65bd')
        self.assertEqual(len(gate['confirmed_class_ids']),6)
        self.assertEqual(gate['config_count'],253)

    def test_01_source_integrity(self):
        a=assert_authorized(); s=run_source_integrity()
        self.assertTrue(a['authorized'])
        self.assertEqual(s['confirmed_class_count'],6)
        self.assertEqual(s['causal_class_count'],1)

    def test_02_pairing_preflight_frozen_before_alignment(self):
        p,s=freeze_pairs()
        self.assertTrue(p['pairing_frozen_before_alignment_results'])
        self.assertEqual(len(s),6)
        self.assertEqual(sum(1 for x in s if x['primary_causal']),1)
        for row in p['pairs']:
            self.assertEqual(row['recipient']['family'],row['same_class_donor']['family'])
            self.assertNotEqual(row['recipient']['organism_id'],row['same_class_donor']['organism_id'])
            self.assertEqual(row['recipient']['family'],row['different_class_donor']['family'])
            self.assertEqual(row['size'],row['different_class_donor']['size'])

    def test_03_legacy_and_identity_baseline_reproduction(self):
        p=reproduce_baselines()
        self.assertTrue(p['identity_reproduced'])
        self.assertTrue(p['legacy_orthogonal_reproduced'])
        self.assertLessEqual(p['max_identity_metric_delta'],1e-9)
        self.assertLessEqual(p['max_legacy_metric_delta'],1e-9)

if __name__=='__main__': unittest.main()
