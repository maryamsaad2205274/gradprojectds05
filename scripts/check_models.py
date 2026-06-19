#!/usr/bin/env python
"""Terminal model health check. Run: py check_models.py"""

from utils.model_health import print_health_report

if __name__ == "__main__":
    raise SystemExit(print_health_report())
