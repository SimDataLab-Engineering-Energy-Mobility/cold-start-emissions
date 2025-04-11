# Copyright (C) 2025 Karlsruhe Institute of Technology (KIT)

# Scientific Computing Center (SCC), Department of Scientific Computing and Mathematics

# Authors: Manoj Mangipudi, Jordan A. Denev

# Licensed under the GNU General Public License v3.0

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)