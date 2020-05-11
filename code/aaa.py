# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as signal
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from util import BaseFunc as bf
from util import BaseChart as bc

a = np.array([1,2,3])
bc.showPlot(a)


