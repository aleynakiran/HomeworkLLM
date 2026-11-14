"""Format actual quantized-student mistakes for inspection after evaluation."""


def failure_report(rows, predictions, limit=10):
    """Return the first mistakes in shared test order, up to the requested limit."""
    if len(rows) != len(predictions):
        raise ValueError('Each test row needs a prediction')
    mistakes = [(row, prediction) for row, prediction in zip(rows, predictions)
                if prediction != row['label']][:limit]

    def escape(value):
        return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')

    lines = ['# Quantized student failure examples', '',
             'The examples below are the first incorrect predictions in shared test-set order.', '']
    if mistakes:
        lines += ['| Message | Locale | Expected intent | Predicted intent |',
                  '| --- | --- | --- | --- |']
        for row, prediction in mistakes:
            lines.append('| ' + ' | '.join(escape(value) for value in
                         (row['text'], row['locale'], row['label'], prediction)) + ' |')
    else:
        lines.append('No incorrect predictions were found.')
    if len(mistakes) < limit:
        lines += ['', 'Fewer incorrect predictions were available than the requested limit.']
    return '\n'.join(lines) + '\n'
