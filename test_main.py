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


def insert_point(data):
    new_record_id = current_data_block.add_record(
                Record_Datafile(
                    point_dim,
                    current_data_block.get_next_available_record_id(),
                    data[0],
                    data[3],
                    [data[1], data[2]]
                )
            )


if __name__ == '__main__':

    # full path to osm file
    osm_path = "C:\\Users\\Public\\Documents\\map.osm"
    # full path to datafile
    datafile_name = 'C:\\Users\\Public\\Documents\\datafile'
    # full path to indexfile
    indexfile_name = 'C:\\Users\\Public\\Documents\\indexfile'

    random.seed(1)


    random_names = [
        'name 1',
        'name 2',
        'name 3',
        'name 4',
        'name 5',
        'name 6',
        'name 7',
        'name 8'
    ]
    osm_read = osm_reader.GetNamesAndLocs()
    osm_read.apply_file(osm_path)
    #elements_to_insert = len(osm_read.ids)
    #elements_to_insert = 20
    
    
    data = osm_read.get_data()

    point_dim = data.shape[1] - 1
    
    elements_to_insert = data.shape[0]

    
    names = []
    for i in range(elements_to_insert):
        names.append(

            random_names[random.randint(0, len(random_names) - 1)]
            
        )
    
    np_names = np.array(names)

    final = np.column_stack((data,np_names))
    
    
    
 


    datafile = open(datafile_name, 'w')
    datafile.close()
    del datafile

    indexfile = open(indexfile_name, 'w')
    indexfile.close()
    del indexfile

    current_data_block = Block_Datafile(
        Block_Datafile.get_next_available_block_id(),
        point_dim,
        struct.calcsize(record_fmt_datafile)
    )
    datafile_blocks_offsets.append(offset_for_next_block)

    offset_for_next_block = \
        block_write_datafile(
            current_data_block,
            datafile_name,
            offset=offset_for_next_block
        )
    
    import time
    start = time.time()
    # Inserting elements one by one takes 2 sec for 100 insertions = 20 ms for each insertion
    for i in range(elements_to_insert):
        if i%100 == 0 :
            print(i)
            
        if current_data_block.is_full():

            block_write_datafile(
                current_data_block,
                datafile_name,
                offset=datafile_blocks_offsets[current_data_block.block_id]
            )

            del current_data_block

            current_data_block = Block_Datafile(
                Block_Datafile.get_next_available_block_id(),
                point_dim,
                struct.calcsize(record_fmt_datafile)
            )

            datafile_blocks_offsets.append(offset_for_next_block)

            offset_for_next_block = block_write_datafile(
                current_data_block,
                datafile_name, 
                offset=offset_for_next_block
            )

            insert_point(final[i])

        else:

            insert_point(final[i])
    end = time.time()
    print(f"Elapsed time: {end-start:.6f} seconds")

    catalog: Rtree = Rtree(index_file_name=indexfile_name)
    catalog.bulk_loading(datafile_name, datafile_blocks_offsets)

    
    # current_data_block.remove_record(15)
    # print(current_data_block.id_to_index)

    # block_write_datafile(
    #     current_data_block,
    #     datafile_name,
    #     offset=datafile_blocks_offsets[current_data_block.block_id]
    # )

    # print(datafile_blocks_offsets)
    # for offset in datafile_blocks_offsets:
        
    #     b: Block_Datafile = block_load_datafile(datafile_name,offset)
    #     print(b)
    #     print(b.id_to_index)
    #     del b

    pass