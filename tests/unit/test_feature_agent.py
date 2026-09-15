"""Unit tests for Feature Agent."""

import pandas as pd
from app.agents.feature_agent import FeatureAgent
from app.knowledge.retriever import KnowledgeRetriever

def test_feature_agent_prompt(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "rules.md").write_text("Calculate risk score as age * income. Do not use SSN.")
    
    cache_dir = tmp_path / ".knowledge"
    
    retriever = KnowledgeRetriever(cache_dir=str(cache_dir))
    retriever.ingest_directory(str(doc_dir))
    
    agent = FeatureAgent()
    agent.retriever = retriever # inject
    
    df = pd.DataFrame({"age": [20], "income": [50000], "target": [1]})
    prompt = agent.run(df, "target", "Banking dataset")
    
    assert "Banking dataset" in prompt
    assert "age, income" in prompt
    assert "target" not in prompt  # target shouldn't be in features list
    assert "Calculate risk score" in prompt

def test_feature_agent_evaluate_gates():
    agent = FeatureAgent()
    
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [50000, 60000, 70000, 80000],
        "target": [0, 1, 0, 1]
    })
    
    new_features = pd.DataFrame({
        "valid_feature": [1, 2, 2, 1],
        "leaky_feature": [0, 1, 0, 1], # perfect predictor of target
        "redundant_feature": [50000, 60000, 70000, 80000], # perfectly correlated with income
        "invalid_feature": [1, 1, 1, 1], # constant
        "invalid_nan": [None, None, None, None] # all NaNs
    })
    
    approved_features = agent.evaluate_proposed_features(df, "target", new_features, is_classification=True)
    
    # Check that only the valid, non-leaky, non-redundant feature passed
    assert list(approved_features.columns) == ["valid_feature"]
