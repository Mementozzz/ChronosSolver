def find_best_answer(target, answers):
    th, tm = target

    for i, (h,m) in enumerate(answers):
        if h == th and m == tm:
            return i
    return None