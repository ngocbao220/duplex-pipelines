from duplexchat_pipe.dialogue import extract_valid_dialogues, split_into_dialogues


def summarize(segments):
    return split_into_dialogues(segments, gap_seconds=5.0), extract_valid_dialogues(segments, gap_seconds=5.0, min_duration_seconds=10.0)
