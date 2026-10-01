import os

if os.environ.get("ATEC_TASK_E_POLICY_MODE", "act").strip().lower() == "rgb":
    from .solution_rgb import AlgSolution
else:
    from .solution_act import AlgSolution

__all__ = ["AlgSolution"]
