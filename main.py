import numpy as np
from datafile_management import Block_Datafile
from datafile_management import Record_Datafile

from r_tree import *

from file_management import *



import struct
import random

from sys import float_info


import osm_reader


#! check again all double loops in list initialisations


datafile_blocks_offsets = []
offset_for_next_block = 0




point_dim = 2
block_size = 2**15



# TODO : check that the formats below are still
# TODO : valid
#! Some formats are given as a string literal at places 
#! in the code, check for these too
rtree_fmt = '>IIIIIII?'

block_fmt_datafile = '>IIII'
block_fmt_indexfile = '>I?'

record_fmt_datafile = '>IQ30s' + ''.join(['d' for _ in range(point_dim)])
record_fmt_indexfile_inner = '>II' + ''.join(['d' for _ in range(2 * point_dim)])
record_fmt_indexfile_leaf = '>III' + ''.join(['d' for _ in range(point_dim)])
    

if __name__ == '__main__':

    # full path to osm file
    osm_path1 = "C:\\Users\\Public\\Documents\\map.osm"
    osm_path2 = "C:\\Users\\Family\\Downloads\\albania-latest.osm.pbf"
    osm_path3 = "C:\\Users\\Family\\Downloads\\malta-latest.osm.pbf"
    # full path to datafile
    datafile_name = 'C:\\Users\\Public\\Documents\\datafile'
    # full path to indexfile
    indexfile_name = 'C:\\Users\\Public\\Documents\\indexfile'

    datafile = open(datafile_name, 'w')
    datafile.close()
    del datafile

    indexfile = open(indexfile_name, 'w')
    indexfile.close()
    del indexfile

    # We define the batch size 
    batch_size = block_size // struct.calcsize(record_fmt_datafile)

    # Calling the handler to move the nodes from osm 
    # in datafile
    handler = osm_reader.BatchHandler(datafile_name, batch_size)

    import time
    start1 = time.time()
    ####################

    handler.apply_file(osm_path1)

    datafile_blocks_offsets = handler.finalize()

    ##########################
    end1 = time.time()
    print(f"Elapsed time for transfer: {end1-start1:.6f} seconds")
    

    
    start2 = time.time()
    #########################

    # Initialization and creation of our catalog
    catalog: Rtree = Rtree(index_file_name=indexfile_name, 
                           maximum_num_of_records= block_size//struct.calcsize(record_fmt_indexfile_leaf),
                           datafile_name_change_that=datafile_name)
    catalog.bulk_loading(datafile_name, datafile_blocks_offsets)



    ###############################
    end2 = time.time()
    print(f"Elapsed time for bulking: {end2-start2:.6f} seconds")

