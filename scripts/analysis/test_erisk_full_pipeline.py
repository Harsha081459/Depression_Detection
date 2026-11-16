import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import erisk_full_pipeline as p


class PipelineChecks(unittest.TestCase):
    def test_spearman_gate_threshold_and_constants(self):
        rows={s:{'n':60,'spearman':.95} for s in p.SIGNALS}
        self.assertTrue(p.pilot_gate_passed(rows))
        for value in (.94999,None,float('nan'),-1.):
            rows['emptiness']['spearman']=value
            self.assertFalse(p.pilot_gate_passed(rows))

    def test_backend_tie_and_gap_diagnostics(self):
        from backend_pilot_gates import comparison_stats,GATE2_MAX_GAP,GATE2_MIN_SPEARMAN
        stats=comparison_stats([0.,0.,.9],[0.,0.,1.])
        self.assertAlmostEqual(stats['hf_exact_tie_rate'],2/3)
        self.assertAlmostEqual(stats['ollama_exact_tie_rate'],2/3)
        self.assertAlmostEqual(stats['max_abs_gap'],.1)
        self.assertAlmostEqual(stats['spearman'],1.)
        self.assertIsNone(comparison_stats([0.,0.],[0.,1.])['spearman'])
        self.assertEqual((GATE2_MAX_GAP,GATE2_MIN_SPEARMAN),(.01,.99))

    def test_synthetic_only_pass_cannot_score_cohort(self):
        scorer=p.LocalScorer.__new__(p.LocalScorer)
        scorer.meta={'fixture':True}; scorer.batch_size=1
        with self.assertRaises(RuntimeError):
            scorer.score(pd.DataFrame(),'unused',{'backend_hash':p.digest(scorer.meta),'passed':True})

    def test_truncation_audit_reports_label_difference(self):
        from erisk_repeated_cv import truncation_audit
        stamp=pd.Timestamp('2020-01-01',tz='UTC')
        posts=pd.DataFrame({'user_id':['a','b'],'date':[stamp,stamp],'text':['x'*2500,'x'*10]})
        audit=truncation_audit(posts,pd.Series({'a':stamp,'b':stamp}),pd.Series({'a':1,'b':0}))
        self.assertTrue((audit.loc[audit.label==1,'truncated_fraction']==1).all())
        self.assertTrue((audit.loc[audit.label==0,'truncated_fraction']==0).all())

    def test_masking_preserves_first_post_origin(self):
        raw=pd.DataFrame({'user_id':['a','a','b'],'post_id':['1','2','3'],
                          'date':pd.to_datetime(['2020-01-01','2020-02-01','2020-01-01'],utc=True),
                          'text':['my depression','Synthetic ordinary writing','my depression']})
        posts,origins=p.prepare_posts(raw)
        self.assertEqual(len(posts),1)
        scores=posts[['user_id','post_id','date']].copy()
        for s in p.SIGNALS: scores[s]=.8
        x=p.window_features(posts,scores,origins,['a','b'],4)
        self.assertEqual(x.n_posts.tolist(),[0.,0.])
        self.assertEqual(x.has_posts.tolist(),[0.,0.])
        later=p.window_features(posts,scores,origins,['a','b'],8)
        self.assertEqual(later.n_posts.tolist(),[1.,0.])

    def test_missing_scores_fail_instead_of_imputing(self):
        raw=pd.DataFrame({'user_id':['a'],'post_id':['1'],
                          'date':pd.to_datetime(['2020-01-01'],utc=True),'text':['Ordinary fixture']})
        posts,origins=p.prepare_posts(raw)
        scores=posts[['user_id','post_id','date']].iloc[:0].copy()
        for s in p.SIGNALS: scores[s]=[]
        with self.assertRaises(AssertionError): p.window_features(posts,scores,origins,['a'],4)

    def test_paired_bootstrap_identical_predictions(self):
        y=np.array([0,0,0,1,1,1]); score=np.array([.1,.2,.8,.3,.7,.9])
        result=p.bootstrap(y,score,other=score)
        self.assertEqual(result,dict(estimate=0.,ci_low=0.,ci_high=0.))

    def test_frozen_split_excludes_pilot_users(self):
        labels=pd.DataFrame({'user_id':[str(i) for i in range(523)],'label':[1]*91+[0]*432})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pilot.parquet'
            labels.iloc[:25][['user_id']].to_parquet(path,index=False)
            a=p.freeze_split(labels,path,Path(d)/'split.json')
            b=p.freeze_split(labels,path,Path(d)/'split.json')
            pd.testing.assert_frame_equal(a,b)
            self.assertTrue((a.loc[labels.user_id.iloc[:25],'split']=='train').all())
            self.assertEqual((a.split=='test').sum(),105)

    def test_selection_ignores_test_labels_and_preserves_controls(self):
        y=np.array([0,1]*15)
        x=pd.DataFrame({'n_posts':1.,'has_posts':1.},index=range(30))
        for s in p.SIGNALS:
            x[s+'__mean']=y
            x[s+'__max']=1-y
        for c in p.CODE: x[c]=0.
        selector=p.SelectSignals('both').fit(x,y)
        self.assertTrue(all(s+'__mean' in selector.columns_ for s in p.SIGNALS))
        self.assertEqual(selector.transform(x).shape,(30,11))


if __name__=='__main__': unittest.main()
