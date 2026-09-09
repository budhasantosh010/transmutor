from __future__ import annotations

import itertools, unittest
import numpy as np
import torch

from experiments.v837_primitive_invention.v837al.adapter_families import fit_vector_map,apply_np,inverse_np
from experiments.v837_primitive_invention.v837al.scope_localization import configurations,complexity
from experiments.v837_primitive_invention.v837al.aligned_primitive import enabled_ports
from experiments.v837_primitive_invention.v837al.canonical_interface import _identity_bundle
from experiments.v837_primitive_invention.v837al.utils import read_json,HERE


class TestV837alTransforms(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(83701);self.x=self.rng.normal(size=(256,4))

    def test_signed_permutation_invertible(self):
        y=self.x[:,[2,0,3,1]]*np.array([-1,1,-1,1]);m=fit_vector_map(self.x,y,'SIGNED_PERMUTATION',require_invertible=True)
        self.assertTrue(m['valid']);self.assertLess(np.max(np.abs(apply_np(self.x,m)-y)),1e-10);self.assertLess(np.max(np.abs(inverse_np(y,m)-self.x)),1e-10)

    def test_diagonal_affine_inverse(self):
        y=self.x*np.array([2.,-.5,1.5,3.])+np.array([1.,2.,-3.,.2]);m=fit_vector_map(self.x,y,'DIAGONAL_AFFINE',require_invertible=True)
        self.assertTrue(m['valid']);self.assertLess(np.max(np.abs(inverse_np(apply_np(self.x,m),m)-self.x)),1e-9)

    def test_rigid_affine_translation_applied_and_orthogonal(self):
        q,_=np.linalg.qr(self.rng.normal(size=(4,4)));y=(self.x-np.array([1.,2.,3.,4.]))@q+np.array([-2.,1.,.5,3.]);m=fit_vector_map(self.x,y,'RIGID_AFFINE',require_invertible=True)
        Q=np.asarray(m['q']);self.assertLess(np.linalg.norm(Q.T@Q-np.eye(4)),1e-9);self.assertGreater(np.linalg.norm(np.asarray(m['mu_y'])-np.asarray(m['mu_x'])),0);self.assertLess(np.max(np.abs(inverse_np(apply_np(self.x,m),m)-self.x)),1e-8)

    def test_full_affine_ridge_and_condition_guard(self):
        A=np.array([[1,.2,0,0],[0,1,.1,0],[0,0,1,.2],[.1,0,0,1.]]);b=np.arange(4.);y=self.x@A+b;m=fit_vector_map(self.x,y,'FULL_AFFINE',require_invertible=True)
        self.assertTrue(m['valid']);self.assertLess(m['diagnostics']['condition_number'],1000);self.assertLess(np.mean((apply_np(self.x,m)-y)**2),1e-12)
        bad=self.x.copy();bad[:,3]=bad[:,2]*1e-9;mb=fit_vector_map(bad,bad,'FULL_AFFINE',require_invertible=True);self.assertFalse(mb['valid'])

    def test_state_roundtrip(self):
        y=self.x*1.3+0.2;m=fit_vector_map(self.x,y,'DIAGONAL_AFFINE',require_invertible=True);self.assertLess(np.max(np.abs(inverse_np(apply_np(self.x,m),m)-self.x)),1e-10)


class TestV837alScope(unittest.TestCase):
    def test_64_unique_port_scopes(self):
        cfg=configurations();self.assertEqual(len({x['scope'] for x in cfg}),64)
    def test_253_unique_configs(self):
        cfg=configurations();self.assertEqual(len(cfg),253);self.assertEqual(len({x['config_id'] for x in cfg}),253)
    def test_identity_config_exact(self):
        c=configurations()[0];self.assertEqual(c['scope'],'000000');self.assertEqual(c['family'],'IDENTITY');self.assertEqual(c['ports'],[])
    def test_scope_bit_order(self):
        self.assertEqual(enabled_ports('100000'),{'state'});self.assertEqual(enabled_ports('000001'),{'output'});self.assertEqual(enabled_ports('111111'),{'state','external_messages','global_term','projected_input','gate','output'})
    def test_complexity_order_deterministic(self):
        for fam in ('SIGNED_PERMUTATION','DIAGONAL_AFFINE','RIGID_AFFINE','FULL_AFFINE'):
            a=complexity(fam,'100000',3);b=complexity(fam,'111111',3);self.assertLessEqual(a['adapter_macs_per_timestep'],b['adapter_macs_per_timestep'])


class TestV837alFrozenData(unittest.TestCase):
    def test_fit_select_test_exact_and_disjoint(self):
        g=read_json(HERE/'frozen_interface_alignment_gate.json');self.assertEqual(g['align_fit_seeds'],[10000,10063]);self.assertEqual(g['align_select_seeds'],[10064,10127]);self.assertEqual(g['align_test_seeds'],[20000,20127]);self.assertTrue(set(range(10000,10064)).isdisjoint(range(10064,10128)))
    def test_no_fresh_audit(self):
        g=read_json(HERE/'frozen_interface_alignment_gate.json');self.assertFalse(g['fresh_audit_consumed']);self.assertFalse(g['v838_started']);self.assertEqual(g['primitives_promoted'],0)


class TestV837alCanonical(unittest.TestCase):
    def test_canonical_anchor_receives_exact_identity(self):
        for family in ('SIGNED_PERMUTATION','DIAGONAL_AFFINE','RIGID_AFFINE','FULL_AFFINE'):
            b=_identity_bundle(family,3)
            self.assertTrue(b['canonical_identity'])
            for port in ('state','external_messages','global_term','output'):
                for m in b['ports'][port]:
                    x=np.arange(12,dtype=float).reshape(3,4)
                    self.assertTrue(np.array_equal(apply_np(x,m),x))
            x6=np.arange(18,dtype=float).reshape(3,6)
            for m in b['ports']['projected_input']:
                self.assertTrue(np.array_equal(apply_np(x6,m),x6))
            self.assertEqual(b['ports']['gate']['a'],1.0);self.assertEqual(b['ports']['gate']['b'],0.0)

if __name__=='__main__':unittest.main()
