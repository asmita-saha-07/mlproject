import logging
import os 
from datetime import datetime
from logging.handlers import RotatingFileHandler

logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path, exist_ok=True)
log_file=os.path.join(logs_path,"app.log")

handler=RotatingFileHandler(log_file, maxBytes=3000, backupCount=3)
formatter=logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s")
handler.setFormatter(formatter)

logger=logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

