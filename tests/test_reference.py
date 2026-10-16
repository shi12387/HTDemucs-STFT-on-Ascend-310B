"""CPU-only analytical checks of padding boundary, normalization and CaC ordering."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_tests import reference,make_cases
from support import baseline
class ReferenceTests(unittest.TestCase):
    def setUp(self):self.window=(.5-.5*np.cos(2*np.pi*np.arange(4096,dtype=np.float64)/4096)).astype(np.float32)
    def test_baseline_sources(self):baseline()
    def test_zero(self):self.assertEqual(np.count_nonzero(reference(np.zeros((2,351232),np.float32),self.window,'numpy')),0)
    def test_cos_sin_planes_and_normalization(self):
        n=np.arange(351232,dtype=np.float64);x=np.stack((np.cos(2*np.pi*5*n/4096),np.sin(2*np.pi*11*n/4096))).astype(np.float32)
        y=reference(x,self.window,'numpy');self.assertEqual(y.shape,(1,4,2048,336))
        # Packed time 2 is source frame 4; its start has integer signal periods.
        self.assertAlmostEqual(float(y[0,0,5,2]),16,places=5);self.assertAlmostEqual(float(y[0,1,5,2]),0,places=5)
        self.assertAlmostEqual(float(y[0,2,11,2]),0,places=5);self.assertAlmostEqual(float(y[0,3,11,2]),-16,places=5)
    def test_shifted_impulse_and_frame_crop(self):
        x=np.zeros((2,351232),np.float32);x[0,3072]=1;y=reference(x,self.window,'numpy')
        self.assertAlmostEqual(float(y[0,0,0,0]),1/128,places=8);self.assertAlmostEqual(float(y[0,1,1,0]),-1/128,places=8)
        self.assertEqual(np.count_nonzero(y[0,2:]),0)
    def test_restore_cases_are_identical(self):
        d=make_cases(2026);self.assertTrue(np.array_equal(d['tones'],d['tones_restore']));self.assertTrue(np.array_equal(d['tones'],d['tones_warmup']))
if __name__=='__main__':unittest.main()
