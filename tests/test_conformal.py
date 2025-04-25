import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResultState, EfficiencyAnalyzer

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
  
