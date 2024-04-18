
from termcolor import colored


LOGGING_INFO  = False
LOGGING_DEBUG = False
LOGGING_ERROR = True


def print_error(msg, args=None):
    if LOGGING_ERROR:
        print_msg('ERROR', msg, 'red')

def print_info(msg, args=None):
    if getattr(args, 'verbose', False) or LOGGING_INFO:
        print_msg('INFO', msg, 'yellow')

def print_debug(msg, args=None):
    if getattr(args, 'debug', False) or LOGGING_INFO:
        print_msg('DEBUG', msg, 'blue')

def print_msg(header, msg, color):
    print(colored(f'[{header}] {msg}', color))
