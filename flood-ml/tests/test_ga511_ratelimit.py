from ga511.ratelimit import RateLimiter, redact


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self.sleeps = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_redact_strips_key_query_param():
    url = "https://511ga.org/api/v2/get/cameras?key=SUPERSECRET123&format=json"
    redacted = redact(url)
    assert "SUPERSECRET123" not in redacted
    assert "key=REDACTED" in redacted
    assert "format=json" in redacted


def test_redact_handles_key_first_param():
    url = "https://511ga.org/api/v2/get/event?key=abc123"
    redacted = redact(url)
    assert "abc123" not in redacted
    assert redacted.endswith("key=REDACTED")


def test_redact_in_exception_message():
    msg = "HTTPSConnectionPool: Max retries exceeded with url: /api/v2/get/cameras?key=zzz&format=json"
    redacted = redact(msg)
    assert "zzz" not in redacted
    assert "key=REDACTED" in redacted


def test_redact_handles_none_and_empty():
    assert redact(None) == ""
    assert redact("") == ""
    assert redact("no key here") == "no key here"


def test_limiter_allows_up_to_max_calls_without_waiting(tmp_path):
    clock = FakeClock()
    limiter = RateLimiter(
        tmp_path / "state.json",
        max_calls=8,
        window_s=60.0,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    for _ in range(8):
        limiter.acquire()
    assert clock.sleeps == []


def test_limiter_blocks_the_9th_call_until_window_clears(tmp_path):
    clock = FakeClock()
    limiter = RateLimiter(
        tmp_path / "state.json",
        max_calls=8,
        window_s=60.0,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    for _ in range(8):
        limiter.acquire()
    start = clock.now
    limiter.acquire()
    assert clock.sleeps, "expected the limiter to sleep before granting the 9th call"
    assert clock.now - start >= 59.5


def test_limiter_is_cross_process_via_shared_state_file(tmp_path):
    clock = FakeClock()
    state_path = tmp_path / "shared_state.json"
    limiter_a = RateLimiter(
        state_path, max_calls=8, window_s=60.0, time_func=clock.time, sleep_func=clock.sleep
    )
    limiter_b = RateLimiter(
        state_path, max_calls=8, window_s=60.0, time_func=clock.time, sleep_func=clock.sleep
    )
    for _ in range(4):
        limiter_a.acquire()
    for _ in range(4):
        limiter_b.acquire()
    assert clock.sleeps == []
    limiter_a.acquire()
    assert clock.sleeps


def test_limiter_raises_timeout_when_max_wait_exceeded(tmp_path):
    clock = FakeClock()
    limiter = RateLimiter(
        tmp_path / "state.json",
        max_calls=1,
        window_s=1000.0,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    limiter.acquire()
    try:
        limiter.acquire(max_wait_s=5.0)
        raised = False
    except TimeoutError:
        raised = True
    assert raised

