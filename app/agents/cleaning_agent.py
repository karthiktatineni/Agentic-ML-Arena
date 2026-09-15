"""Data Cleaning Agent — LLM-driven with rule-based fallback (PRD v2 Section 12)."""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CleaningAgent:
    """Analyzes a DataFrame, asks the LLM for a cleaning plan, then executes it."""

    def __init__(self, llm_provider=None, run_id: str = "global"):
        self.llm = llm_provider
        self.run_id = run_id

    def _publish_degraded_mode_event(self, reason: str):
        from app.events.schemas import DegradedModeEvent
        from app.events.bus import event_bus

        event_bus.publish(DegradedModeEvent(
            run_id=self.run_id,
            subsystem="CleaningAgent",
            reason=reason,
        ))

    def _trip_degraded_mode(self, reason: str):
        from app.governance.llm_governor import global_governor

        if global_governor.trip(reason):
            self._publish_degraded_mode_event(global_governor.trip_reason)

    # ------------------------------------------------------------------ #
    #  DATA PROFILING (input to the LLM)
    # ------------------------------------------------------------------ #
    def _profile(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Build a compact data profile the LLM can reason about."""
        profile: Dict[str, Any] = {
            "rows": len(df),
            "columns": len(df.columns),
            "target": target,
            "column_profiles": [],
        }
        for col in df.columns:
            info: Dict[str, Any] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_pct": round(df[col].isnull().mean() * 100, 1),
                "nunique": int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                info["mean"] = round(float(desc.get("mean", 0)), 4)
                info["std"] = round(float(desc.get("std", 0)), 4)
                info["min"] = round(float(desc.get("min", 0)), 4)
                info["max"] = round(float(desc.get("max", 0)), 4)
                # skewness
                try:
                    info["skew"] = round(float(df[col].skew()), 4)
                except Exception:
                    pass
            profile["column_profiles"].append(info)
        return profile

    # ------------------------------------------------------------------ #
    #  LLM CLEANING PLAN
    # ------------------------------------------------------------------ #
    def _ask_llm_for_plan(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send profile to LLM and get a structured cleaning plan."""
        if self.llm is None:
            return None

        from app.governance.llm_governor import global_governor

        if global_governor.check_governor():
            logger.info(f"LLMGovernor tripped; falling back to deterministic rules: {global_governor.trip_reason}")
            return None

        system = (
            "You are an expert ML data engineer. Given a dataset profile, "
            "return a JSON cleaning plan. The JSON must have these keys:\n"
            '  "drop_columns": [list of column names to drop],\n'
            '  "impute_numeric": {"column": "strategy"} where strategy is "median"|"mean"|"zero",\n'
            '  "impute_categorical": {"column": "strategy"} where strategy is "mode"|"unknown",\n'
            '  "cap_outliers": [list of column names to cap at 3 sigma],\n'
            '  "drop_duplicates": true|false,\n'
            '  "reasoning": "short explanation"\n'
            "Return ONLY valid JSON, no markdown."
        )
        prompt = json.dumps(profile, indent=2)

        try:
            resp = self.llm.generate(prompt, system_prompt=system, json_mode=True, temperature=0.1)
            was_tripped = global_governor.check_governor()
            if getattr(resp, "cost_usd", 0) > 0:
                global_governor.record_cost(resp.cost_usd)
                if not was_tripped and global_governor.check_governor():
                    self._publish_degraded_mode_event(global_governor.trip_reason)
                
            if resp.success and resp.content:
                try:
                    plan = json.loads(resp.content)
                    global_governor.record_success()
                    logger.info(f"LLM cleaning plan: {plan.get('reasoning', '')}")
                    return plan
                except json.JSONDecodeError:
                    was_tripped = global_governor.check_governor()
                    global_governor.record_schema_error()
                    if not was_tripped and global_governor.check_governor():
                        self._publish_degraded_mode_event(global_governor.trip_reason)
                    logger.warning("LLM returned invalid JSON schema.")
            else:
                reason = resp.error or "LLM provider returned an unsuccessful empty response."
                self._publish_degraded_mode_event(f"CleaningAgent LLM failure: {reason}")
                global_governor.trip(reason)
                logger.warning(f"LLM cleaning plan failed, falling back to rules: {reason}")
        except Exception as e:
            self._publish_degraded_mode_event(f"CleaningAgent LLM exception: {e}")
            global_governor.trip(str(e))
            logger.warning(f"LLM cleaning plan failed, falling back to rules: {e}")
        return None

    # ------------------------------------------------------------------ #
    #  RULE-BASED FALLBACK
    # ------------------------------------------------------------------ #
    def _rule_based_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic cleaning rules when LLM is unavailable."""
        plan: Dict[str, Any] = {
            "drop_columns": [],
            "impute_numeric": {},
            "impute_categorical": {},
            "cap_outliers": [],
            "drop_duplicates": True,
            "reasoning": "Rule-based fallback (LLM unavailable)",
        }
        target = profile["target"]
        for cp in profile["column_profiles"]:
            name = cp["name"]
            if name == target:
                continue
            # Drop columns with >60% nulls
            if cp["null_pct"] > 60:
                plan["drop_columns"].append(name)
                continue
            # Drop constant columns
            if cp["nunique"] <= 1:
                plan["drop_columns"].append(name)
                continue
            # Impute remaining nulls
            if cp["null_pct"] > 0:
                if cp["dtype"] in ("float64", "int64", "float32", "int32"):
                    plan["impute_numeric"][name] = "median"
                else:
                    plan["impute_categorical"][name] = "mode"
            # Cap extreme skew
            skew = cp.get("skew")
            if skew is not None and abs(skew) > 3:
                plan["cap_outliers"].append(name)
        return plan

    # ------------------------------------------------------------------ #
    #  EXECUTE PLAN
    # ------------------------------------------------------------------ #
    def _execute_plan(self, df: pd.DataFrame, plan: Dict[str, Any], target: str) -> pd.DataFrame:
        """Apply the cleaning plan to the DataFrame."""
        result = df.copy()

        # 0. Upstream data sanitization: unit stripping, synonym canonicalization, sentinels, CarAge
        synonym_maps = {
            "transmission": {
                "at": "automatic",
                "auto": "automatic",
                "automatic": "automatic",
                "mt": "manual",
                "man": "manual",
                "manual": "manual",
            },
            "fueltype": {
                "diesel": "diesel",
                "petrol": "petrol",
                "cng": "cng",
                "lpg": "lpg",
                "electric": "electric",
            },
            "fuel": {
                "diesel": "diesel",
                "petrol": "petrol",
                "cng": "cng",
                "lpg": "lpg",
                "electric": "electric",
            },
            "ownercount": {
                "1": 1, "1st": 1, "first": 1, "first owner": 1,
                "2": 2, "2nd": 2, "second": 2, "second owner": 2,
                "3": 3, "3rd": 3, "third": 3, "third owner": 3,
                "4": 4, "4th": 4, "fourth": 4, "4th & above": 4, "fourth & above": 4,
                "test drive car": 0,
            },
            "owner": {
                "1": 1, "1st": 1, "first": 1, "first owner": 1,
                "2": 2, "2nd": 2, "second": 2, "second owner": 2,
                "3": 3, "3rd": 3, "third": 3, "third owner": 3,
                "4": 4, "4th": 4, "fourth": 4, "4th & above": 4, "fourth & above": 4,
                "test drive car": 0,
            },
        }

        for col in result.columns:
            if col == target:
                continue
            col_key = col.lower().replace("_", "").replace(" ", "")

            # If object/string, perform whitespace stripping, lowercase, synonym mapping, and regex unit-stripping
            if result[col].dtype == "object" or result[col].dtype.name == "string":
                clean_str = result[col].astype(str).str.strip().str.lower()
                clean_str = clean_str.replace("nan", np.nan).replace("none", np.nan).replace("", np.nan)

                # Check canonical synonym mapping
                if col_key in synonym_maps:
                    mapped = clean_str.map(lambda v: synonym_maps[col_key].get(v, v) if pd.notna(v) else v)
                    # Check if mapped values are ordinal numbers (e.g. OwnerCount)
                    num_try = pd.to_numeric(mapped, errors="coerce")
                    if num_try.notna().sum() / max(1, clean_str.notna().sum()) >= 0.7:
                        result[col] = num_try
                        logger.info(f"Cleaning: canonicalized ordinal '{col}' to numeric")
                    else:
                        result[col] = mapped
                        logger.info(f"Cleaning: canonicalized categorical synonyms in '{col}'")
                else:
                    # Explicit regex unit-stripping (e.g. '1005 CC', '11.5 kmpl', '150614 km')
                    extracted = clean_str.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
                    num_extracted = pd.to_numeric(extracted, errors="coerce")
                    valid_ratio = num_extracted.notna().sum() / max(1, clean_str.notna().sum())
                    if valid_ratio >= 0.7:
                        result[col] = num_extracted
                        logger.info(f"Cleaning: converted unit-contaminated string '{col}' to numeric")
                    else:
                        result[col] = clean_str

            # Null out negative sentinels on naturally positive columns (e.g. EngineCC=-1197, KM < 0)
            if pd.api.types.is_numeric_dtype(result[col]):
                if any(k in col_key for k in ["engine", "km", "mileage", "distance", "price", "year", "owner"]):
                    invalid_mask = result[col] < 0
                    if invalid_mask.any():
                        result.loc[invalid_mask, col] = np.nan
                        logger.info(f"Cleaning: nulled out negative sentinels in '{col}'")

        # Derive CarAge domain feature if Year is present
        year_col = next((c for c in result.columns if c.lower() == "year" and c != target), None)
        if year_col and "CarAge" not in result.columns:
            current_yr = 2026
            result["CarAge"] = current_yr - pd.to_numeric(result[year_col], errors="coerce")
            logger.info(f"Cleaning: derived domain feature 'CarAge' from '{year_col}'")

        # 1. Drop columns
        drop_cols = [c for c in plan.get("drop_columns", []) if c in result.columns and c != target]
        if drop_cols:
            logger.info(f"Cleaning: dropping columns {drop_cols}")
            result = result.drop(columns=drop_cols)

        # 2. Drop duplicates
        if plan.get("drop_duplicates", True):
            before = len(result)
            result = result.drop_duplicates()
            logger.info(f"Cleaning: dropped {before - len(result)} duplicate rows")

        # 3. Impute numeric
        for col, strategy in plan.get("impute_numeric", {}).items():
            if col not in result.columns:
                continue
            if strategy == "median":
                result[col] = result[col].fillna(result[col].median())
            elif strategy == "mean":
                result[col] = result[col].fillna(result[col].mean())
            elif strategy == "zero":
                result[col] = result[col].fillna(0)

        # 4. Impute categorical
        for col, strategy in plan.get("impute_categorical", {}).items():
            if col not in result.columns:
                continue
            if strategy == "mode":
                mode_val = result[col].mode()
                result[col] = result[col].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")
            else:
                result[col] = result[col].fillna("UNKNOWN")

        # 5. Cap outliers at 3σ
        for col in plan.get("cap_outliers", []):
            if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
                continue
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                lower = mean - 3 * std
                upper = mean + 3 * std
                result[col] = result[col].clip(lower, upper)
                logger.info(f"Cleaning: capped outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")

        return result

    # ------------------------------------------------------------------ #
    #  PUBLIC API
    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the full cleaning pipeline. Returns (cleaned_df, plan_used)."""
        profile = self._profile(df, target)

        # Try LLM first, fall back to rules
        plan = self._ask_llm_for_plan(profile)
        is_llm = True
        if plan is None:
            plan = self._rule_based_plan(profile)
            is_llm = False

        cleaned = self._execute_plan(df, plan, target)
        
        from app.events.schemas import AgentDecisionEvent
        from app.events.bus import event_bus
        
        drop_c = len(plan.get("drop_columns", []))
        cap_c = len(plan.get("cap_outliers", []))
        imp_n = len(plan.get("impute_numeric", {}))
        imp_c = len(plan.get("impute_categorical", {}))
        
        action = f"Dropped {drop_c} cols, capped {cap_c}, imputed {imp_n + imp_c} cols."
        event_bus.publish(AgentDecisionEvent(
            run_id=self.run_id,
            agent_name="CleaningAgent",
            decision_action=action,
            confidence=0.92 if is_llm else 1.0,
            reasoning_summary=plan.get("reasoning", "Applied cleaning plan")
        ))
        
        return cleaned, plan
