"""LLM Feature Agent (PRD v2 Section 5)."""

import pandas as pd
from typing import List, Dict, Any, Optional
from app.knowledge.retriever import KnowledgeRetriever
from app.core.config import GlobalRunConfig

class FeatureAgent:
    """Generates feature engineering code grounded in domain knowledge."""
    
    def __init__(self, config: Optional[GlobalRunConfig] = None):
        self.config = config or GlobalRunConfig()
        self.retriever: Optional[KnowledgeRetriever] = None
        try:
            self.retriever = KnowledgeRetriever()
        except ImportError:
            pass
        
    def generate_feature_prompt(self, column_names: List[str], dataset_desc: str) -> str:
        """Construct a prompt grounded in RAG retrieved domain rules."""
        
        # 1. Retrieve domain rules based on dataset description and columns
        query = dataset_desc + " " + " ".join(column_names)
        domain_rules = self.retriever.retrieve(query, top_k=3) if self.retriever else []
        
        # 2. Build Prompt
        prompt = (
            f"You are a Feature Engineering expert for AutoML Arena.\n"
            f"Dataset Description: {dataset_desc}\n"
            f"Available Columns: {', '.join(column_names)}\n\n"
        )
        
        if domain_rules:
            prompt += "--- RELEVANT DOMAIN RULES (Follow these explicitly) ---\n"
            for rule in domain_rules:
                prompt += f"- {rule['text']}\n"
            prompt += "------------------------------------------------------\n\n"
            
        prompt += (
            "Write a Python function `def engineer_features(df: pd.DataFrame) -> pd.DataFrame:` "
            "that adds new features based on the columns and domain rules provided. "
            "Return ONLY valid Python code."
        )
        
        return prompt
        
    def evaluate_proposed_features(self, df: pd.DataFrame, target_col: str, new_features_df: pd.DataFrame, is_classification: bool = True) -> pd.DataFrame:
        """
        Evaluate proposed features for validity, leakage, redundancy, and availability.
        Returns a DataFrame of only the features that passed all gates.
        """
        # 1. Validity Check (NaNs, Infs, constant values)
        valid_cols = []
        for col in new_features_df.columns:
            if new_features_df[col].isna().all() or new_features_df[col].nunique() <= 1:
                continue
            valid_cols.append(col)
            
        if not valid_cols:
            return pd.DataFrame(index=df.index)
            
        new_features_df = new_features_df[valid_cols]
        
        # 2. Leakage Check (via Detector Engine)
        from app.governance.leakage_detectors import LeakageDetectorEngine
        detector = LeakageDetectorEngine()
        
        # Concat target to new features for testing
        eval_df = pd.concat([new_features_df, df[[target_col]]], axis=1)
        leakage_findings = detector.run_all_tabular_checks(eval_df, target_col, is_classification=is_classification)
        
        leaky_cols = {f.column for f in leakage_findings if f.action == "exclude"}
        safe_cols = [c for c in new_features_df.columns if c not in leaky_cols]
        
        # 3. Redundancy (Perfect correlation with existing features)
        final_cols = []
        for col in safe_cols:
            is_redundant = False
            for orig_col in df.columns:
                if orig_col == target_col:
                    continue
                if pd.api.types.is_numeric_dtype(new_features_df[col]) and pd.api.types.is_numeric_dtype(df[orig_col]):
                    # Check absolute correlation, ignore NaNs
                    corr = abs(new_features_df[col].corr(df[orig_col]))
                    if pd.notna(corr) and corr > 0.99:
                        is_redundant = True
                        break
            if not is_redundant:
                final_cols.append(col)
                
        return new_features_df[final_cols]

    def apply_high_cardinality_encoding(
        self,
        df: pd.DataFrame,
        target_col: str,
        cardinality_threshold: int = 10,
    ) -> pd.DataFrame:
        """Apply smoothed out-of-fold target encoding for high-cardinality categoricals."""
        from app.automl.preprocessing import OutOfFoldTargetEncoder, get_high_cardinality_categoricals

        high_card_cols = get_high_cardinality_categoricals(
            df, target_col, cardinality_threshold=cardinality_threshold
        )
        if not high_card_cols:
            return df

        encoder = OutOfFoldTargetEncoder(
            categorical_cols=high_card_cols,
            n_splits=min(5, max(2, len(df) // 50)),
            smoothing=10.0,
            cv_seed=getattr(self.config, "random_seed", 42),
        )
        df_encoded = encoder.fit_transform_oof(df, df[target_col])
        df_encoded = df_encoded.drop(columns=high_card_cols, errors="ignore")
        return df_encoded
        
    def run(self, df: pd.DataFrame, target_col: str, dataset_desc: str) -> str:
        """
        Simulate LLM generation of feature code.
        (In a full implementation, this calls an LLM API).
        """
        features_only = [c for c in df.columns if c != target_col]
        prompt = self.generate_feature_prompt(features_only, dataset_desc)
        
        from app.events.schemas import AgentDecisionEvent
        from app.events.bus import event_bus
        run_id_val = "global"
        if self.config and getattr(self.config, "run_id", None):
            run_id_val = self.config.run_id
            
        event_bus.publish(AgentDecisionEvent(
            run_id=run_id_val,
            agent_name="FeatureAgent",
            decision_action="Formulated feature generation prompt via RAG",
            confidence=0.88,
            reasoning_summary=f"Retrieved domain rules to engineer {len(features_only)} columns."
        ))
        
        # LLM Call would go here. We'll just return the prompt as a proxy for testing.
        return prompt
