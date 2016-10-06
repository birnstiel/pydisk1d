#!/usr/bin/env python
"""
Merges hdf5 data files from several runs based on the given order and the file name.
"""
import glob, os, shutil, argparse, sys
from cStringIO import StringIO
from pydisk1D import pydisk1D

# to make loading / saving the data quiet

class Capturing(list):
    """
    Context that captures the standard output of a function
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
    
def get_lists(bases=['parts1','parts2','parts3_crashed'],mask='hdf*/*/*.hdf5'):
    """
    Get list of lists of files
    
    first list is everything that matches bases[0]+'/'+mask
    second one is everything that matches bases[1]+'/'+mask
    and so on.
    """
    return [glob.glob(os.path.join(base,mask)) for base in bases]
    
def merge(lists,skip=1,outdir='merge'):
    """
    Merges lists of simulations. Simulations in the second list are setups that continue
    the simulations of the first list. Every file name that is equal in a subset or all of
    the lists is merged and written out into the output directory.
    
    Arguments:
    ----------
    
    lists : list of lists
    :   each element of this list is itself a list of pathes to simulation data sets
    
    Keywords:
    ---------

    skip : int
    :   how many elements to skip on each merge, default: 1 to avoid overlap
    """
    
    initial_files = lists.pop(0)
    
    if os.path.isdir(outdir):
        raise OSError('output directory exists, please remove it.')
    else:
        os.mkdir(outdir)
        
    print('Merging data ...')
    
    for i_file,initial_file in enumerate(initial_files):
    
        # get all matching files for this one
        
        merge_list = []
        for list in lists:
            for item in list:
                if os.path.basename(initial_file)==os.path.basename(item):
                    merge_list += [item]
        
        with Capturing() as output:
        
            # read initial object
        
            d = pydisk1D(initial_file)
        
            for merge_file in merge_list:
                m = pydisk1D(merge_file)
                d.join(m,skip=skip)
            
            # store merged object
        
            fname = os.path.splitext(os.path.basename(initial_file))[0]
            d.data_dir = fname
            d.save_diskev()
            shutil.move(fname+'.hdf5',os.path.join(outdir,fname+'.hdf5'))
        
        # print progress
        sys.stdout.write("\r{}: {:2.2%}".format(fname,(i_file+1.)/len(initial_files)))
        sys.stdout.flush()
        
if __name__=='__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('-m','--mask',   help='mask which to apply after each base to list all hdf5 files in that base directory', type=str,default='*')
    PARSER.add_argument('first',         help='initial directory',            type=str,metavar='directory1')
    PARSER.add_argument('second',        help='further base directories',     type=str,nargs='+',metavar='directory')
    PARSER.add_argument('-o','--outdir', help='output directory',             type=str,default='merge')
    PARSER.add_argument('-s','--skip',   help='number of skipped time steps', type=int,default=1)
    
    ARGS = PARSER.parse_args()
    bases = [ARGS.first]+ARGS.second
    mask = ARGS.mask
    
    merge(get_lists(bases,mask),skip=ARGS.skip,outdir=ARGS.outdir)