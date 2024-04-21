#!/usr/bin/env python3

def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
