import React from 'react'
import { BookOpen, AlertCircle, CheckCircle2, ShieldAlert } from 'lucide-react'

export const ExplanationMethodology: React.FC = () => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 bg-blue-500/10 text-blue-400 rounded-lg border border-blue-500/20">
          <BookOpen className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-white">Scientific Attribution Methodology & Non-Causal Boundaries</h3>
          <p className="text-xs text-slate-400">Mathematical foundation of model interpretability and empirical safety rules</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-slate-800/40 p-4 rounded-lg border border-slate-700/60">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" /> 1. Global Interpretability
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            Compares <strong>Model-Native Gini Impurity</strong> (split frequency across ensemble decision trees) against{' '}
            <strong>Out-of-Sample Permutation Importance</strong> evaluated strictly on holdout test partitions (2016–2017).
            Methodological rank disagreements are explicitly surfaced rather than concealed.
          </p>
        </div>

        <div className="bg-slate-800/40 p-4 rounded-lg border border-slate-700/60">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" /> 2. Local Prediction Attribution
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            Decomposes an individual prediction via <strong>Marginal Reference Perturbation</strong>. Evaluates the shift in predicted
            yield when substituting single feature values into the empirical dataset median baseline reference vector.
          </p>
        </div>

        <div className="bg-slate-800/40 p-4 rounded-lg border border-slate-700/60">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" /> 3. Controlled Sensitivity Analysis
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            Executes deterministic [-10%, -5%, 0%, +5%, +10%] parameter sweeps on active inputs while strictly enforcing
            physical non-negative domain boundaries for acreage and yield parameters.
          </p>
        </div>

        <div className="bg-slate-800/40 p-4 rounded-lg border border-slate-700/60">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" /> 4. Decision Traceability & Audit Certificates
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            Every explanation generates an immutable <strong>SHA-256 certificate (EXP-xxxx)</strong> capturing the exact model version,
            dataset provenance, input vector, and attribution weights for complete historical auditability.
          </p>
        </div>
      </div>

      <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-200/90 leading-relaxed">
          <strong className="text-amber-300">Mandatory Scientific Integrity Directive:</strong> Explanations describe the machine learning
          model's internal decision boundaries and statistical loss gradients within the trained ICRISAT feature distribution. They must{' '}
          <strong>never</strong> be interpreted as physical, agronomic, or causal intervention guarantees.
        </div>
      </div>
    </div>
  )
}
