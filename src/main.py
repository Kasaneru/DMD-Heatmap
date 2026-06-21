from solver import solve
from heatdmd import heat_dmd
from tester import dmd_diagnose
import os
import sys
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    conf_name = sys.argv[1]
    if conf_name:
        solve(conf_name, to_show=True)
        heat_dmd(conf_name, to_show=True)
        dmd_diagnose(conf_name)
    else:
        print('Введите имя файла конфигурации без расширения!')

