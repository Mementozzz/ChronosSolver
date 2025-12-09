def find_best_answer(target, answers):
    """Find the closest matching answer to the target time"""
    th, tm = target

    # First try exact match
    for i, (h, m) in enumerate(answers):
        if h == th and m == tm:
            return i, 0  # Return index and difference
    
    # If no exact match, find the closest one
    min_diff = float('inf')
    best_idx = None
    
    for i, (h, m) in enumerate(answers):
        # Convert both to total minutes for comparison
        target_total = th * 60 + tm
        answer_total = h * 60 + m
        
        # Calculate difference (handle wraparound at 12 hours)
        diff = abs(target_total - answer_total)
        
        # Handle wraparound case (e.g., 11:50 is close to 0:10)
        if diff > 360:  # More than 6 hours difference
            diff = 720 - diff  # 12 hours = 720 minutes
        
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    
    # Only return if the closest match is within 10 minutes
    if best_idx is not None and min_diff <= 10:
        return best_idx, min_diff
    
    return None, None