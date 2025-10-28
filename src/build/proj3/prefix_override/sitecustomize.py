import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/paurush/CSCE752/CSCE752_Project3/src/install/proj3'
