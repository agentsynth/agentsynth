from agentsynth.schemas import Trajectory, EvalResult
from agentsynth.metrics import filter_dataset

def test_filter_dataset():
    t1 = Trajectory(id="1", query="test1", mode="single_agent")
    t2 = Trajectory(id="2", query="test2", mode="multi_agent", verification={"verified": True})
    t3 = Trajectory(id="3", query="test3", mode="code_execution", verification={"verified": False})
    
    evals = [
        EvalResult(trajectory_id="1", overall=0.5),
        EvalResult(trajectory_id="2", overall=0.8),
        EvalResult(trajectory_id="3", overall=0.9),
    ]
    
    # 1. Filter by score
    kept, report = filter_dataset([t1, t2, t3], evals, min_score=0.6)
    assert len(kept) == 2
    assert kept[0].id == "2"
    assert kept[1].id == "3"
    assert report["dropped_score"] == 1
    
    # 2. Filter by verification
    kept, report = filter_dataset([t1, t2, t3], evals, min_score=0.0, verified_only=True)
    assert len(kept) == 1
    assert kept[0].id == "2"
    assert report["dropped_verification"] == 2
    
    # 3. Filter by mode
    kept, report = filter_dataset([t1, t2, t3], evals, min_score=0.0, modes=["multi_agent"])
    assert len(kept) == 1
    assert kept[0].id == "2"
    assert report["dropped_mode"] == 2
