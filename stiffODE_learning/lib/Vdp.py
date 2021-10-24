# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 05:22:03 2021

@author: LLSCAU
"""

def VDPode(t, y):
    k1 = 0.25
    k2 = 1.0
    y1, y2 = y
    #rhs
    dy1dt = k1 * y1
    dy2dt = k2 * (y2 - y1**2)
    return dy1dt, dy2dt
    