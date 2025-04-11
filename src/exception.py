# Copyright (C) 2025 Karlsruhe Institute of Technology (KIT)

# Scientific Computing Center (SCC), Department of Scientific Computing and Mathematics

# Authors: Manoj Mangipudi, Jordan A. Denev

# Licensed under the GNU General Public License v3.0

import sys
import logging
from logger import logger

def error_message_detail(error, error_detail:sys):
    _, _, exe_tb=error_detail.exc_info() # info about error in the code: which file, which line etc 
    file_name=exe_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name, exe_tb.tb_lineno, str(error)    
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message_detail