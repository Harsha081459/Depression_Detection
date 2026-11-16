import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import erisk_repeated_cv as cv
from erisk_full_pipeline import SIGNALS


class SearchFixture:
    fits=0
    allowed_sizes={418,419}
    def __init__(self,*args,**kwargs): pass
    def fit(self,X,y):
        self.train=set(X.index)
        assert len(self.train) in SearchFixture.allowed_sizes
        assert X.index.equals(y.index)
        self.best_params_={}; SearchFixture.fits+=1
        return self
    def predict_proba(self,X):
        assert not self.train.intersection(X.index)
        value=X.iloc[:,0].to_numpy()
        return np.column_stack([1-value,value])


class RepeatedCVChecks(unittest.TestCase):
    def test_outer_folds_are_disjoint_and_cover_523_users_five_times(self):
        ids=pd.Index(['fixture_'+str(i) for i in range(523)])
        y=pd.Series([1]*91+[0]*432,index=ids)
        X=pd.DataFrame({s+'__mean':(np.arange(523)%17)/17 for s in SIGNALS},index=ids)
        def select(train,labels,signal):
            self.assertEqual(len(train),len(labels))
            self.assertTrue(train.index.equals(labels.index))
            self.assertLess(len(train),523)
            return signal+'__mean',{}
        SearchFixture.fits=0
        SearchFixture.allowed_sizes={418,419}
        with tempfile.TemporaryDirectory() as tmp,patch.object(cv,'GridSearchCV',SearchFixture),patch.object(cv,'choose_feature',side_effect=select):
            out=Path(tmp)/'cv'
            result=cv.repeated_cv({4:X},y,out)
            self.assertTrue((result.folds==25).all())
            self.assertEqual(SearchFixture.fits,25*8)
            pred=pd.read_parquet(out/'out_of_fold_predictions_PRIVATE.parquet')
            counts=pred.groupby(['user_id','model','ablation']).size()
            self.assertTrue((counts==5).all())
            with self.assertRaises(RuntimeError): cv.repeated_cv({4:X},y,out)

    def test_without_pilot_uses_only_unexposed_users(self):
        ids=pd.Index(['fixture_'+str(i) for i in range(523)])
        y=pd.Series([1]*91+[0]*432,index=ids)
        pilot=ids[:20].union(ids[100:140]); keep=ids.difference(pilot)
        X=pd.DataFrame({s+'__mean':(np.arange(523)%17)/17 for s in SIGNALS},index=ids).loc[keep]
        SearchFixture.allowed_sizes={370,371}; SearchFixture.fits=0
        def select(train,labels,signal):
            self.assertFalse(set(train.index).intersection(pilot))
            return signal+'__mean',{}
        with tempfile.TemporaryDirectory() as tmp,patch.object(cv,'GridSearchCV',SearchFixture),patch.object(cv,'choose_feature',side_effect=select):
            out=Path(tmp)/'cv_without'
            result=cv.repeated_cv({4:X},y.loc[keep],out,population='without_pilot')
            self.assertTrue((result.users==463).all())
            self.assertTrue((result.folds==25).all())
            pred=pd.read_parquet(out/'out_of_fold_predictions_PRIVATE.parquet')
            self.assertFalse(set(pred.user_id).intersection(pilot))


if __name__=='__main__': unittest.main()
