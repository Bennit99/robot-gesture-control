import time

iterations = []
names = []
last_timestamp = 0


def start_next_iteration(name: str):
    global iterations, last_timestamp
    if len(iterations) == 0:
        # first call, only take timestamp
        last_timestamp = time.perf_counter()
    else:
        time_me(name)
    iterations.append([])


def time_me(name: str):
    global last_timestamp, iterations
    if len(iterations) == 1:  # first iteration
        global names
        names.append(name)
    iterations[-1].append(time.perf_counter() - last_timestamp)
    last_timestamp = time.perf_counter()


def average_times() -> dict[str, float]:
    global iterations, names
    #  names
    print("Names: " + str(names))

    # count and print dimensions of iterations
    lengths = {}
    for it in iterations:
        length = len(it)
        lengths[length] = lengths.get(length, 0) + 1
    print("Iterations lengths count:", lengths)

    # drop last iteration if it's incomplete (e.g. due to early exit)
    if len(iterations) > 0 and len(iterations[-1]) < len(names):
        iterations = iterations[:-1]

    # Calculate averages carefully to avoid zip truncation if lengths differ
    averages = {}
    for i, name in enumerate(names):
        times = [it[i] for it in iterations if i < len(it)]
        if times:
            averages[name] = (sum(times) * 1000) / len(times)
    return averages


def reset_time_tracker():
    global iterations, names, last_timestamp
    iterations = []
    names = []
    last_timestamp = 0
