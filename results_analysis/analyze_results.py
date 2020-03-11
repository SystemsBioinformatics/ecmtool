import pandas as pd
from helpers_analyze_results import *
import numpy as np

old_inout_ecms = pd.read_csv('KO8_old_inout_conversion_cone.csv', header=None).values
new_inout_ecms = pd.read_csv('KO8_new_inout_conversion_cone.csv', header=None).values

bijection_YN, ecms_first_min_ecms_second, ecms_second_min_ecms_first = check_bijection_csvs(
    np.transpose(old_inout_ecms), np.transpose(new_inout_ecms))

pass
