"""Champion Report Generation (PRD v2 Section 61)."""

import pandas as pd
from typing import Dict, Any, List
from app.api.schemas.experiment import ExperimentObject

class ChampionReportGenerator:
    """Generates the Human-in-the-loop Champion Report."""
    
    @staticmethod
    def _calculate_disparate_impact(df: pd.DataFrame, y_pred: pd.Series, protected_col: str) -> Dict[str, float]:
        """Calculate Disparate Impact for a protected attribute."""
        if protected_col not in df.columns:
            return {}
            
        results = {}
        df_eval = df.copy()
        df_eval['pred'] = y_pred
        
        # Binary prediction assumed for classification
        overall_positive_rate = df_eval['pred'].mean()
        
        for category in df_eval[protected_col].unique():
            cat_mask = df_eval[protected_col] == category
            if not cat_mask.any():
                continue
                
            cat_positive_rate = df_eval.loc[cat_mask, 'pred'].mean()
            
            # Disparate impact ratio (category rate / overall rate)
            # Usually DI < 0.8 is considered problematic (80% rule)
            di = cat_positive_rate / overall_positive_rate if overall_positive_rate > 0 else 1.0
            results[str(category)] = di
            
        return results
    
    @classmethod
    def generate_report(
        cls, 
        experiment: ExperimentObject, 
        eval_df: pd.DataFrame, 
        eval_preds: pd.Series,
        leakage_status: str,
        degraded_mode: bool
    ) -> str:
        """Generate the comprehensive Markdown report."""
        
        metrics = experiment.metrics
        protected_attrs = experiment.dataset_protected_attributes
        
        lines = []
        lines.append(f"# Champion Approval Report: {experiment.run_id}")
        lines.append(f"**Model Family**: {experiment.model_family}")
        lines.append(f"**Experiment Hash**: {experiment.experiment_hash}")
        lines.append(f"**Degraded Mode Triggered**: {'Yes (Search Space Constrained)' if degraded_mode else 'No'}")
        lines.append(f"**Leakage Audit Status**: {leakage_status}")
        lines.append("")
        
        lines.append("## 1. Performance")
        if metrics:
            lines.append(f"- **Primary Metric ({metrics.primary_metric})**: {metrics.mean_cv_score:.4f} ± {metrics.std_cv_score:.4f}")
            lines.append(f"- **Selection-Validation Score**: {metrics.selection_val_score if metrics.selection_val_score else 'N/A'}")
        lines.append("")
        
        lines.append("## 2. Fairness & Bias Analysis")
        if not protected_attrs:
            lines.append("No protected attributes were identified during ingestion.")
        else:
            lines.append("Disparate Impact Analysis (80% rule: DI < 0.8 indicates potential bias):")
            for attr in protected_attrs:
                di_results = cls._calculate_disparate_impact(eval_df, eval_preds, attr)
                lines.append(f"\n### Protected Attribute: {attr}")
                if not di_results:
                    lines.append("- *Attribute not found in evaluation dataset.*")
                for cat, di in di_results.items():
                    flag = " ⚠️ (Action Required)" if di < 0.8 else ""
                    lines.append(f"- **{cat}**: DI = {di:.2f}{flag}")
                    
        lines.append("\n## 3. Human Approval Gate")
        lines.append("To deploy this model to the prediction API, an explicit `POST /api/approval/approve` request is required with this `run_id`.")
        
        return "\n".join(lines)
