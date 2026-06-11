from app.ratelimit import RateLimiter


def test_allows_up_to_limit():
    rl = RateLimiter(limit=3, window_seconds=3600)
    assert all(rl.allow("1.2.3.4", now=100.0 + i) for i in range(3))
    assert rl.allow("1.2.3.4", now=104.0) is False
    assert rl.allow("5.6.7.8", now=104.0) is True  # other IPs unaffected


def test_window_slides():
    rl = RateLimiter(limit=2, window_seconds=60)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=1.0) is True
    assert rl.allow("ip", now=2.0) is False
    assert rl.allow("ip", now=62.0) is True  # the first hit aged out


def test_rejected_attempts_do_not_consume_quota():
    rl = RateLimiter(limit=1, window_seconds=60)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=1.0) is False
    assert rl.allow("ip", now=61.0) is True  # only the accepted hit counted
