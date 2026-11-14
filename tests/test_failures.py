from src.eval.failures import failure_report


def test_failure_report_selects_actual_mistakes_and_escapes_markdown():
    rows = [{'text': f'message | {i}', 'locale': 'en-US', 'label': 'correct'} for i in range(12)]
    predictions = ['correct'] + ['wrong'] * 11
    report = failure_report(rows, predictions)
    assert 'message \\| 0' not in report
    assert 'message \\| 1' in report
    assert 'message \\| 10' in report
    assert 'message \\| 11' not in report


def test_failure_report_handles_no_mistakes():
    report = failure_report([{'text': 'hello', 'locale': 'en-US', 'label': 'x'}], ['x'])
    assert 'No incorrect predictions were found.' in report
