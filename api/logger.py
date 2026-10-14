"""Structured JSON logging with request_id, latency, and full audit replay support."""
import json
import logging
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms
        if hasattr(record, "route"):
            log_record["route"] = record.route
        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code
        if hasattr(record, "extra_data"):
            log_record["data"] = record.extra_data

        return json.dumps(log_record, ensure_ascii=False)

def setup_logger():
    logger = logging.getLogger("salary_api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger

api_logger = setup_logger()
