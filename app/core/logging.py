import json
import logging
import sys


logger = logging.getLogger("mica")
logger.setLevel(logging.INFO)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {"time": self.formatTime(record), "level": record.levelname, "message": record.getMessage()}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


handler = logging.StreamHandler(sys.stdout)
formatter = JsonFormatter()
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)
