def format_time_gap(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"+{minutes}:{remaining_seconds:.3f}"
    else:
        return f"+{remaining_seconds:.3f}"
