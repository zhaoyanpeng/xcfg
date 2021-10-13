# -*- coding: utf-8 -*-
import os 
import sys
import logging
import json
import re

import numpy as np 

from collections import defaultdict, Counter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Operand(object):
    
    def __init__(self, idx):
        self.idx = idx 

class FloatOperand(Operand):
    
    def __init__(self, idx):
        super(FloatOperand, self).__init__(idx)

class GaussianMxitureOperand(Operand):
    
    def __init__(self, idx):
        super(GaussianMixtureOperand, self).__init__(idx)

