import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e191_physical_support as P


class PhysicalSupportTests(unittest.TestCase):
    def track(self):
        t=np.array([0,30,60,90,105,120,150,180,210,240.],dtype=float)
        distance=np.array([0,0,0,0,20,80,200,400,600,800.])
        return [t,50+distance/111200,np.full(10,10.),np.array([0,0,0,0,2,3,5,10,10,10.]),np.ones(10,bool),np.zeros(10,bool),300.,(50.,10.)]
    def test_observable_bracket_yields_interval_and_midpoint(self):
        r=P.observe(*self.track());self.assertEqual(r['reason'],'eligible')
        self.assertEqual(r['q_mid'],202.5);self.assertEqual(r['bracket_s'],15.)
        self.assertFalse(r['return_to_stand'])
    def test_queue_far_from_stand_is_not_offblock(self):
        a=self.track();a[-1]=(51.,10.);self.assertNotEqual(P.observe(*a)['reason'],'eligible')
    def test_large_gap_near_movement_is_rejected(self):
        a=self.track();a[0][3]=65;self.assertNotEqual(P.observe(*a)['reason'],'eligible')
    def test_return_to_stand_is_flagged_and_excluded(self):
        a=self.track();a[1][-1]=50.;a[3][-1]=0.;r=P.observe(*a)
        self.assertTrue(r['return_to_stand']);self.assertNotEqual(r['reason'],'eligible')
    def test_missing_speed_or_stale_dwell_does_not_become_stationary(self):
        for k,v in [(3,np.nan),(5,True)]:
            a=self.track();a[k][3]=v;self.assertNotEqual(P.observe(*a)['reason'],'eligible')
    def test_unknown_stand_and_old_track_rejected(self):
        a=self.track();a[-1]=None;self.assertEqual(P.observe(*a)['reason'],'unknown_stand')
        a=self.track();a[-2]=1000.;self.assertEqual(P.observe(*a)['reason'],'no_recent_ground')
    def test_other_flight_on_same_aircraft_is_not_the_departure(self):
        a=self.track();callsigns=np.array(['ABC1']+['OTHER']*9)
        self.assertEqual(P.observe(*a,callsigns=callsigns,expected_callsign='ABC1')['reason'],'conflicting_segment_callsign')
        callsigns=np.array([None]*9+['ABC1'],dtype=object)
        self.assertEqual(P.observe(*a,callsigns=callsigns,expected_callsign='ABC1')['reason'],'eligible')
    def test_duplicate_times_cannot_claim_zero_uncertainty(self):
        a=self.track();a[0][4]=90.
        self.assertEqual(P.observe(*a)['reason'],'duplicate_segment_timestamps')


if __name__=='__main__':unittest.main()
