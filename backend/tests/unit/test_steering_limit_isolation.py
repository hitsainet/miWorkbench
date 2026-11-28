"""Each test starts with a steering rate limiter no earlier test has spent (tests/conftest.py,
`_each_test_its_own_steering_limit`).

The two tests below run in file order on one worker (`--dist loadfile`; no random-order plugin is installed): the
first spends the limit, the second must find it unspent. Removing the conftest fixture turns the second red. Why
it matters: CI run 43983730350 (2026-11-17) failed the gate backstop's row 32 with a 429 the tests before it had
earned, while every local run passed.
"""
from __future__ import annotations

CLIENT = "127.0.0.1"


def test_a_test_spends_the_steering_limit():
    from src.api.v1.endpoints import steering

    for _ in range(steering.RATE_LIMIT_REQUESTS):
        assert steering._rate_limiter.is_allowed(CLIENT)
    assert not steering._rate_limiter.is_allowed(CLIENT), "the limit is spent for the rest of this test"


def test_the_next_test_finds_the_steering_limit_unspent():
    from src.api.v1.endpoints import steering

    assert steering._rate_limiter.is_allowed(CLIENT), "a 429 earned by the test before this one"
