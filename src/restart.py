#!/usr/bin/env python
"""
Script to restart diskevolution simulations.
"""
import pydisk1D
import argparse

PARSER = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
PARSER.add_argument('-i',         help='snapshot index to use for restarting, default: last',  type=int,default=-1)
PARSER.add_argument('simulation', help='HDF5 file(s) and/or simulation folders',type=str,nargs='+')
ARGS = PARSER.parse_args()

it = ARGS.i

for sim in ARGS.simulation:
    d=pydisk1D.pydisk1D(i)
    d.write_setup(it,overwrite=True)
