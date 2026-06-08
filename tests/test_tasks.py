from agentsynth.tasks import SEED_TASKS, domains, sample_tasks


def test_seed_bank_spans_domains():
    assert len(SEED_TASKS) >= 20
    assert len(domains()) >= 5
    assert "coding" in domains()


def test_sample_is_deterministic():
    a = sample_tasks(5, seed=3)
    b = sample_tasks(5, seed=3)
    assert [t.id for t in a] == [t.id for t in b]
    assert len(a) == 5


def test_sample_respects_domain_filter():
    picked = sample_tasks(4, domains=["coding"], seed=1)
    assert picked and all(t.domain == "coding" for t in picked)


def test_sample_cycles_when_n_exceeds_pool():
    picked = sample_tasks(100, domains=["coding"], seed=0)
    assert len(picked) == 100
