import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResultState, EfficiencyAnalyzer
from expectation.conformal.adaptivethreshold import AdaptiveThresholdHandler
from expectation.conformal import conformal

class DummyMixture:
    def __init__(self, v, alpha): pass
    def log_superMG(self, s, v): return 0.0  # log(1) = 0

def test_conformal_evalue_init_and_reset(monkeypatch):
    # Patch mixture to avoid dependency
    monkeypatch.setattr(conformal, "OneSidedNormalMixture", DummyMixture)
    e = conformal.ConformalEValue()
    assert e.n_samples == 0
    e._n_samples = 5
    e.reset()
    assert e.n_samples == 0

def test_conformal_evalue_update_and_score(monkeypatch):
    monkeypatch.setattr(conformal, "OneSidedNormalMixture", DummyMixture)
    e = conformal.ConformalEValue()
    val = e.update([1.0, 2.0, 3.0])
    assert val > 0
    assert e.n_samples == 3

def test_conformal_evalue_invalid_type():
    with pytest.raises(ValueError):
        conformal.ConformalEValue(nonconformity_type="invalid")

def test_conformal_evalue_infinite(monkeypatch):
    class InfMixture:
        def __init__(self, v, alpha): pass
        def log_superMG(self, s, v): return np.inf
    monkeypatch.setattr(conformal, "OneSidedNormalMixture", InfMixture)
    e = conformal.ConformalEValue()
    with pytest.raises(ValueError):
        e.update([1.0])

def test_pseudomartingale_update_and_reset():
    pm = conformal.ConformalEPseudomartingale()
    cap, max_cap = pm.update(2.0)
    assert cap == 2.0
    assert max_cap == 2.0
    pm.reset()
    assert pm.capital == pm.initial_capital
    assert pm.max_capital == pm.initial_capital

def test_pseudomartingale_compound_bet():
    pm = conformal.ConformalEPseudomartingale(initial_capital=2.0)
    prod = pm.compound_bet([2.0, 0.5])
    assert prod == 2.0

def test_pseudomartingale_infinite():
    pm = conformal.ConformalEPseudomartingale(allow_infinite=False)
    with pytest.raises(ValueError):
        pm.update(np.inf)

def test_pseudomartingale_test_threshold():
    pm = conformal.ConformalEPseudomartingale()
    pm.update(5.0)
    assert pm.test_threshold(5.0)
    assert not pm.test_threshold(10.0)

def test_pseudomartingale_get_history():
    pm = conformal.ConformalEPseudomartingale()
    pm.update(2.0)
    e_hist, c_hist = pm.get_history()
    assert len(e_hist) == 1
    assert len(c_hist) == 2

def test_truncated_pseudomartingale_truncates():
    pm = conformal.TruncatedEPseudomartingale(min_capital=1.0)
    pm.update(0.0)
    assert pm.capital == 1.0

def test_initialization_defaults():
    handler = AdaptiveThresholdHandler()
    assert handler.target_rate == 0.05
    assert handler.learning_rate == 0.01
    assert handler.threshold == 10.0
    assert handler.min_threshold == 1.0
    assert handler.max_threshold == 50.0
    assert handler.cumulative_error == 0.0
    assert handler.alarm_history == []

def test_update_increases_threshold_when_below_target():
    handler = AdaptiveThresholdHandler(target_false_alarm_rate=0.2)
    old_threshold = handler.threshold
    # current_rate < target_rate, so threshold should increase
    new_threshold = handler.update(current_rate=0.1)
    assert new_threshold > old_threshold

def test_update_decreases_threshold_when_above_target():
    handler = AdaptiveThresholdHandler(target_false_alarm_rate=0.05)
    old_threshold = handler.threshold
    # current_rate > target_rate, so threshold should decrease
    new_threshold = handler.update(current_rate=0.2)
    assert new_threshold < old_threshold

def test_threshold_never_below_min_or_above_max():
    handler = AdaptiveThresholdHandler(min_threshold=5.0, max_threshold=15.0)
    handler.threshold = 10.0
    # Force decrease below min
    for _ in range(20):
        handler.update(current_rate=1.0)  # Should decrease
    assert handler.threshold >= handler.min_threshold
    # Force increase above max
    handler.threshold = 10.0
    for _ in range(20):
        handler.update(current_rate=0.0)  # Should increase
    assert handler.threshold <= handler.max_threshold

def test_record_alarm_and_get_current_rate():
    handler = AdaptiveThresholdHandler()
    handler.record_alarm(True)
    handler.record_alarm(False)
    handler.record_alarm(True)
    assert handler.alarm_history == [True, False, True]
    assert handler.get_current_rate() == 2/3

def test_get_current_rate_empty_history():
    handler = AdaptiveThresholdHandler()
    assert handler.get_current_rate() == 0.0

def test_reset_functionality():
    handler = AdaptiveThresholdHandler()
    handler.record_alarm(True)
    handler.threshold = 20.0
    handler.cumulative_error = 5.0
    handler.reset()
    assert handler.alarm_history == []
    assert handler.threshold == 10.0
    assert handler.cumulative_error == 0.

def test_cusum_initialization_and_reset():
    detector = ConformalCUSUM(threshold=5)
    assert detector.threshold == 5
    detector._cusum_stat = 10
    detector.reset()
    assert detector._cusum_stat == 0.0
    assert detector._alarms == []

def test_cusum_update_triggers_alarm():
    detector = ConformalCUSUM(threshold=2)
    result = detector.update(3)
    assert result.n_alarms == 1
    assert result.alarms == [1]
    assert result.alarm_stats[0] >= 2

def test_cusum_update_no_alarm():
    detector = ConformalCUSUM(threshold=10)
    for _ in range(5):
        result = detector.update(0.5)
    assert result.n_alarms == 0

def test_cusum_truncate_behavior():
    detector = ConformalCUSUM(threshold=10, truncate=True, min_value=1e-3)
    result = detector.update(0.0)
    assert result.statistic == 1e-3

def test_get_alarm_rate():
    detector = ConformalCUSUM(threshold=1.5)
    detector.update(2)
    detector.update(2)
    assert detector.get_alarm_rate() == 1/2

def test_efficiency_analyzer_likelihood_ratios():
    analyzer = EfficiencyAnalyzer()
    data = np.array([0, 0.5, 1.0])
    ratios = analyzer.compute_likelihood_ratios(data)
    assert ratios.shape == data.shape

def test_efficiency_analyzer_decay_short():
    analyzer = EfficiencyAnalyzer()
    decay, se = analyzer.analyze_decay([1], 0)
    assert decay == 0.0
    assert se == float('inf')
  
