"""EDA Agent — LLM-driven exploratory data analysis with rule-based fallback (PRD v2 Section 11)."""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class EDAAgent:
    """Analyzes a DataFrame and produces actionable insights for downstream stages."""

    def __init__(self, llm_provider=None, run_id: str = "global"):
        self.llm = llm_provider
        self.run_id = run_id

    def _publish_degraded_mode_event(self, reason: str):
        from app.events.schemas import DegradedModeEvent
        from app.events.bus import event_bus

        event_bus.publish(DegradedModeEvent(
            run_id=self.run_id,
            subsystem="EDAAgent",
            reason=reason,
        ))

    def _trip_degraded_mode(self, reason: str):
        from app.governance.llm_governor import global_governor

        if global_governor.trip(reason):
            self._publish_degraded_mode_event(global_governor.trip_reason)

    # ------------------------------------------------------------------ #
    #  STATISTICAL PROFILE
    # ------------------------------------------------------------------ #
    def _compute_stats(self, df: pd.DataFrame, target: str, is_classification: bool) -> Dict[str, Any]:
        """Compute comprehensive statistics about the dataset."""
        stats: Dict[str, Any] = {
            "shape": list(df.shape),
            "target": target,
            "task_type": "classification" if is_classification else "regression",
            "null_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        # Target distribution
        if is_classification:
            vc = df[target].value_counts()
            stats["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            stats["class_imbalance_ratio"] = round(float(vc.max() / vc.min()), 2) if vc.min() > 0 else float("inf")
        else:
            stats["target_mean"] = round(float(df[target].mean()), 4)
            stats["target_std"] = round(float(df[target].std()), 4)
            stats["target_skew"] = round(float(df[target].skew()), 4)

        # Feature correlations with target
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if target in numeric_cols:
            correlations = {}
            for col in numeric_cols:
                if col != target:
                    corr = df[col].corr(df[target])
                    if pd.notna(corr):
                        correlations[col] = round(float(corr), 4)
            # Top 5 most correlated
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            stats["top_correlations"] = dict(sorted_corrs)

        # Skewed features
        skewed = []
        for col in numeric_cols:
            if col != target:
                try:
                    s = float(df[col].skew())
                    if abs(s) > 2:
                        skewed.append({"column": col, "skew": round(s, 2)})
                except Exception:
                    pass
        stats["highly_skewed_features"] = skewed

        # High cardinality categoricals
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        high_card = [{"column": c, "nunique": int(df[c].nunique())}
                     for c in cat_cols if df[c].nunique() > 50]
        stats["high_cardinality_categoricals"] = high_card

        return stats

    # ------------------------------------------------------------------ #
    #  LLM INSIGHTS
    # ------------------------------------------------------------------ #
    def _ask_llm_for_insights(self, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send stats to LLM and get actionable insights."""
        if self.llm is None:
            return None

        from app.governance.llm_governor import global_governor

        if global_governor.check_governor():
            logger.info(f"LLMGovernor tripped; falling back to deterministic EDA: {global_governor.trip_reason}")
            return None

        system = (
            "You are an expert ML data scientist performing EDA. Given dataset statistics, "
            "return a JSON object with these keys:\n"
            '  "insights": [list of key observations as strings],\n'
            '  "recommendations": {\n'
            '    "handle_imbalance": "none"|"smote"|"class_weights"|"undersample",\n'
            '    "scaling_needed": true|false,\n'
            '    "log_transform_columns": [list of column names that would benefit from log transform],\n'
            '    "drop_high_cardinality": [list of column names to consider dropping]\n'
            "  }\n"
            "Return ONLY valid JSON, no markdown."
        )
        prompt = json.dumps(stats, indent=2, default=str)

        try:
            resp = self.llm.generate(prompt, system_prompt=system, json_mode=True, temperature=0.1)
            was_tripped = global_governor.check_governor()
            if getattr(resp, "cost_usd", 0) > 0:
                global_governor.record_cost(resp.cost_usd)
                if not was_tripped and global_governor.check_governor():
                    self._publish_degraded_mode_event(global_governor.trip_reason)

            if resp.success and resp.content:
                try:
                    insights = json.loads(resp.content)
                    global_governor.record_success()
                    logger.info(f"LLM EDA insights: {len(insights.get('insights', []))} observations")
                    return insights
                except json.JSONDecodeError:
                    was_tripped = global_governor.check_governor()
                    global_governor.record_schema_error()
                    if not was_tripped and global_governor.check_governor():
                        self._publish_degraded_mode_event(global_governor.trip_reason)
                    logger.warning("LLM returned invalid EDA JSON schema.")
            else:
                reason = resp.error or "LLM provider returned an unsuccessful empty response."
                self._publish_degraded_mode_event(f"EDAAgent LLM failure: {reason}")
                global_governor.trip(reason)
                logger.warning(f"LLM EDA failed, falling back to rules: {reason}")
        except Exception as e:
            self._publish_degraded_mode_event(f"EDAAgent LLM exception: {e}")
            global_governor.trip(str(e))
            logger.warning(f"LLM EDA failed, falling back to rules: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  RULE-BASED FALLBACK
    # ------------------------------------------------------------------ #
    def _rule_based_insights(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic EDA insights when LLM is unavailable."""
        insights: List[str] = []
        recs: Dict[str, Any] = {
            "handle_imbalance": "none",
            "scaling_needed": False,
            "log_transform_columns": [],
            "drop_high_cardinality": [],
        }

        insights.append(f"Dataset has {stats['shape'][0]} rows and {stats['shape'][1]} columns")
        insights.append(f"Task type: {stats['task_type']}")

        if stats.get("null_total", 0) > 0:
            insights.append(f"Found {stats['null_total']} total null values across all columns")

        if stats.get("duplicate_rows", 0) > 0:
            insights.append(f"Found {stats['duplicate_rows']} duplicate rows")

        # Imbalance
        imbalance = stats.get("class_imbalance_ratio", 1)
        if imbalance > 5:
            insights.append(f"Severe class imbalance detected (ratio {imbalance}:1). Recommending class weights.")
            recs["handle_imbalance"] = "class_weights"
        elif imbalance > 2:
            insights.append(f"Moderate class imbalance detected (ratio {imbalance}:1).")
            recs["handle_imbalance"] = "class_weights"

        # Skew
        for s in stats.get("highly_skewed_features", []):
            insights.append(f"Column '{s['column']}' is highly skewed (skew={s['skew']})")
            recs["log_transform_columns"].append(s["column"])

        if recs["log_transform_columns"]:
            recs["scaling_needed"] = True

        # High cardinality
        for hc in stats.get("high_cardinality_categoricals", []):
            insights.append(f"Column '{hc['column']}' has high cardinality ({hc['nunique']} unique values)")
            if hc["nunique"] > 200:
                recs["drop_high_cardinality"].append(hc["column"])

        return {"insights": insights, "recommendations": recs}

    # ------------------------------------------------------------------ #
    #  PUBLIC API
    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame, target: str, is_classification: bool) -> Dict[str, Any]:
        """Run the full EDA pipeline. Returns insights and recommendations."""
        stats = self._compute_stats(df, target, is_classification)

        result = self._ask_llm_for_insights(stats)
        if result is None:
            result = self._rule_based_insights(stats)

        # Merge raw stats into the result for downstream use
        result["stats"] = stats

        from app.events.schemas import AgentDecisionEvent
        from app.events.bus import event_bus

        insights_summary = "; ".join(result.get("insights", [])[:3])
        event_bus.publish(AgentDecisionEvent(
            run_id=self.run_id,
            agent_name="EDAAgent",
            decision_action=f"Profiled {stats.get('shape', [0, 0])[0]} rows. Generated {len(result.get('insights', []))} insights.",
            confidence=0.94,
            reasoning_summary=insights_summary or "Computed summary statistics, distributions, and correlation maps."
        ))

        return result
