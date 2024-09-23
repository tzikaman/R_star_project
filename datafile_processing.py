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
    osm_path = "C:\\Users\\Public\\Documents\\map.osm"
    # full path to datafile
    datafile_name = 'C:\\Users\\Public\\Documents\\datafile'
    # full path to indexfile
    indexfile_name = 'C:\\Users\\Public\\Documents\\indexfile'

    random.seed(1)

    elements_to_insert = 2000

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
    elements_to_insert = len(osm_read.ids)
    
    
    data = []
    print("start of transfering data to main program")
    for i in range(elements_to_insert):
        data.append(
            [
                osm_read.ids[i], 
                osm_read.lats[i], 
                osm_read.lons[i], 
                random_names[random.randint(0, len(random_names) - 1)]
            ]
        )
    print("end")
    
    datafile = open(datafile_name, 'w')
    datafile.close()
    del datafile

    indexfile = open(indexfile_name, 'w')
    indexfile.close()
    del indexfile
    
    catalog: Rtree = Rtree(
        index_file_name=indexfile_name
    )

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
    

    # Inserting elements one by one takes 2 sec for 100 insertions = 20 ms for each insertion
    for i in range(elements_to_insert):

        #! debug start
        if i == 2527:
            pass
        #! debug end

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

            ###################################################################
            new_record_id = current_data_block.add_record(
                Record_Datafile(
                    point_dim,
                    current_data_block.get_next_available_record_id(),
                    data[i][0],
                    data[i][3],
                    [data[i][1], data[i][2]]
                )
            )

            catalog.insert_point(
                [current_data_block.block_id, new_record_id],
                [data[i][1], data[i][2]]
            )
            ###################################################################

        else:
            ###################################################################
            new_record_id = current_data_block.add_record(
                Record_Datafile(
                    point_dim,
                    current_data_block.get_next_available_record_id(),
                    data[i][0],
                    data[i][3],
                    [data[i][1], data[i][2]]
                )
            )

            catalog.insert_point(
                [current_data_block.block_id, new_record_id],
                [data[i][1], data[i][2]]
            )
            ##################################################################

    block_write_datafile(
        current_data_block,
        datafile_name,
        offset=datafile_blocks_offsets[current_data_block.block_id]
    )

    rtree_write_indexfile(
        catalog,
        indexfile_name
    )

    first_datafile_block = None
    catalog = None

    catalog = rtree_read_indexfile(
        indexfile_name
    )

    pass