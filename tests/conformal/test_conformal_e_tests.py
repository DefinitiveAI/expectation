import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResultState, EfficiencyAnalyzer
from expectation.conformal.adaptivethreshold import AdaptiveThresholdHandler

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
  
