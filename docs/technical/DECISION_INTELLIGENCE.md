# Decision Intelligence & Scenario Simulation Architecture

## 1. Decision Workspace Architecture

Implemented in `src/decision_workspace.py` and visualized in the React frontend (`frontend/src/pages/DecisionWorkspace.tsx`):
- **Decision Brief Synthesis**: Combines point forecast, historical district mean, autoregressive lags, empirical uncertainty bounds, feature attributions, and drift alerts into an inspectable summary.
- **Role**: Operates strictly as human decision support, not autonomous policy execution.

---

## 2. Bounded Scenario Simulation

- **Separation Invariant**: All simulation outputs are explicitly tagged as **`[SCENARIO]`** to prevent confusion with empirical forecasts (**`[PREDICTED]`**) or ground truth (**`[OBSERVED]`**).
- **Manifold Bounding**: Parameter shifts are bounded to the 5th–95th historical percentiles, issuing warnings if inputs exceed historical district extremes.
- **Multi-Objective Optimization**: Employs SLSQP optimization (`scipy.optimize.minimize`) to evaluate regional crop mix trade-offs under land and water constraints.
