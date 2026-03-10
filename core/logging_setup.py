import logging

logging.basicConfig(
    filename='logs_ameso.log',  # The file where logs will be written
    filemode='a',            # 'a' for append (add to existing file), 'w' for overwrite
    level=logging.INFO,      # The minimum level of messages to log (e.g., INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger_aemso = logging.getLogger('AEMSOlogs')
