import pytest
from src.trend_analysis import trend_analysis_engine
from backend.services.trend_service import trend_service

def test_trend_engine_increasing_series():
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    yields = [3000, 3100, 3250, 3400, 3550, 3700, 3850, 4000]
    res = trend_analysis_engine.analyze_series(years, yields)
    assert res['theil_sen_slope'] > 100.0
    assert res['direction'] in ["INCREASING", "STRONG INCREASING"]
    assert res['p_value'] < 0.05
    assert res['total_change_pct'] > 30.0

def test_trend_engine_stable_series():
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    yields = [3000, 3010, 2995, 3005, 3000, 3015, 2990, 3002]
    res = trend_analysis_engine.analyze_series(years, yields)
    assert abs(res['theil_sen_slope']) < 10.0
    assert res['direction'] == "STABLE"

def test_trend_service_state_trends():
    trends = trend_service.get_all_states_trends()
    assert len(trends) >= 18
    punjab = next((s for s in trends if s['state'] == 'Punjab'), None)
    assert punjab is not None
    assert 'theil_sen_slope' in punjab
    assert 'direction' in punjab
