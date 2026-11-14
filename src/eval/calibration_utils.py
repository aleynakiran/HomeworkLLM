"""Select a confidence cutoff using validation out-of-scope detection."""
import math


def choose_threshold(confidences, is_out_of_scope):
    """Maximize the mean of in-scope acceptance and out-of-scope rejection.

    Ties favor the lower threshold. Only distinct validation confidences need
    to be considered: rejection happens strictly below the cutoff.
    """
    if len(confidences) != len(is_out_of_scope) or not confidences:
        raise ValueError('Confidence and label lists must be nonempty and aligned')
    if not all(0 <= confidence <= 1 for confidence in confidences):
        raise ValueError('Confidence must be between zero and one')
    in_count = sum(not out for out in is_out_of_scope)
    out_count = sum(is_out_of_scope)
    if not in_count or not out_count:
        raise ValueError('Calibration needs both in-scope and out-of-scope validation examples')
    # Sweep sorted scores once. Equal scores must cross the cutoff together.
    pairs = sorted(zip(confidences, is_out_of_scope))
    best_threshold, best_score = 0.0, 0.5
    in_accepted, out_rejected = in_count, 0
    index = 0
    while index < len(pairs):
        confidence = pairs[index][0]
        if confidence == 1:
            break  # A cutoff cannot exceed one.
        while index < len(pairs) and pairs[index][0] == confidence:
            if pairs[index][1]:
                out_rejected += 1
            else:
                in_accepted -= 1
            index += 1
        score = (in_accepted / in_count + out_rejected / out_count) / 2
        if score > best_score:
            best_threshold = math.nextafter(confidence, math.inf)
            best_score = score
    return best_threshold
