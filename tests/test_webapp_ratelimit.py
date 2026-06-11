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


def test_cleanup_drops_stale_keys():
    rl = RateLimiter(limit=2, window_seconds=60)
    rl.allow("old-ip", now=0.0)
    rl.allow("fresh-ip", now=50.0)
    rl.cleanup_expired(now=90.0)
    assert "old-ip" not in rl._hits
    assert "fresh-ip" in rl._hits


def test_cleanup_keeps_active_keys_quota_intact():
    rl = RateLimiter(limit=1, window_seconds=60)
    assert rl.allow("ip", now=0.0) is True
    rl.cleanup_expired(now=30.0)
    assert rl.allow("ip", now=31.0) is False  # quota still consumed
