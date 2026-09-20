"""
Scenario Audit Service.

Manages execution history, audit logs, and certificate retrieval for simulated scenarios.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.scenario_audit import scenario_audit_engine


class ScenarioAuditService:
    _instance: Optional['ScenarioAuditService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ScenarioAuditService, cls).__new__(cls)
            cls._instance._audit_store: Dict[str, Dict[str, Any]] = {}
            cls._instance._history: List[Dict[str, Any]] = []
        return cls._instance

    def record_scenario_execution(self, audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """Stores a scenario audit record in memory."""
        sc_id = audit_record['scenario_id']
        self._audit_store[sc_id] = audit_record
        # Keep latest 50 records in history
        self._history.insert(0, audit_record)
        if len(self._history) > 50:
            self._history.pop()
        return audit_record

    def get_audit_record(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific scenario audit record by ID."""
        return self._audit_store.get(scenario_id)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent scenario executions."""
        return self._history[:limit]


scenario_audit_service = ScenarioAuditService()
