from datafile_management import *
from indexfile_management import *


import struct


datafile_blocks_offsets = []
offset_for_next_block = 0

########################################################################################## temporary solution
# full path to osm file
osm_path = "C:\\Users\\Public\\Documents\\map.osm"
# full path to datafile
datafile_name = 'C:\\Users\\Public\\Documents\\datafile'
# full path to indexfile
indexfile_name = 'C:\\Users\\Public\\Documents\\indexfile'
##########################################################################################


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



# It writes a specified block with all its contents in the specified
# offset of the specified file and returns where the file pointer stopped 

def block_write_datafile(b: Block_Datafile, datafile_name, offset):

    with open(datafile_name, 'r+b') as datafile:
        datafile.seek(offset)
        packed_block = struct.pack(
            block_fmt_datafile, 
            b.block_id,
            b.block_size, 
            b.size,
            b.record_id_counter
        )
        datafile.write(packed_block)

        for i in range(b.max_num_of_records):
            args = [b.records[i].record_id, 
                    b.records[i].id, 
                    b.records[i].name.encode('utf-8')] + \
                        [b.records[i].vec[j] for j in range(point_dim)]
            packed_record = struct.pack(record_fmt_datafile, *args)
            datafile.write(packed_record)

        new_offset_for_next_block = datafile.tell()

        datafile.flush()

        return new_offset_for_next_block


def block_write_indexfile(b, indexfile_name, offset) -> int:

    with open(indexfile_name, 'r+b') as indexfile:
        indexfile.seek(offset)

        # Write the block
        packed_block = struct.pack(block_fmt_indexfile, b.size, b.is_leaf)
        indexfile.write(packed_block)

        # Write the records of the block
        for i in range(b.max_num_of_records):
            if b.is_leaf:
                args = [b.records[i].record_id, 
                        b.records[i].datafile_record_stored[0], 
                        b.records[i].datafile_record_stored[1]] + \
                            [b.records[i].vec[j] for j in range(point_dim)] # it is a leaf node therefore it is a single point
                packed_record = struct.pack(record_fmt_indexfile_leaf, *args)
                indexfile.write(packed_record)
            else:
                args = [b.records[i].record_id, 
                        b.records[i].datafile_record_stored] + \
                    [extreme_value for dimension in b.records[i].vec for extreme_value in dimension] # it is an MBR because its not a leaf

                packed_record = struct.pack(record_fmt_indexfile_inner, *args)
                indexfile.write(packed_record)
        new_offset_for_next_block_to_enter = indexfile.tell()
        

        indexfile.flush()
        
        return new_offset_for_next_block_to_enter
    
def record_load_datafile(r_id: int, b_id: int) -> Record_Datafile:
    block: Block_Datafile = block_load_datafile(datafile_name, datafile_blocks_offsets[b_id])
    record = block.get_record(r_id)

    return record


            
def block_load_datafile(datafile_name, offset) -> Block_Datafile:

    with open(datafile_name, 'r+b') as datafile:
        datafile.seek(offset)

        # Unpack block 
        data_read = datafile.read(struct.calcsize(block_fmt_datafile))

        unpacked_block = struct.unpack(block_fmt_datafile, data_read)

        b: Block_Datafile = Block_Datafile(
            block_id=unpacked_block[0],
            point_dim=point_dim,
            size_of_record=struct.calcsize(record_fmt_datafile), 
            block_size=unpacked_block[1], 
            size=unpacked_block[2],
            record_id_counter=unpacked_block[3]
        )

        # Unpack records
        for i in range(b.max_num_of_records):
            data_read = datafile.read(struct.calcsize(record_fmt_datafile))
            unpacked_record = struct.unpack(record_fmt_datafile, data_read)
            b.records[i] = Record_Datafile(
                        point_dim, 
                        record_id=unpacked_record[0], 
                        id=unpacked_record[1], 
                        name=unpacked_record[2]
                        .decode('utf-8')
                        .replace('\0', ''), 
                        vec=[
                            unpacked_record[i] \
                                for i in range(3, len(unpacked_record))
                        ]
                    )
            # this whole thing is confusing
            if i < b.size:
                b.id_to_index[unpacked_record[0]] = i 


        return b


def block_load_indexfile(indexfile_name, offset, b_id):

    with open(indexfile_name, 'r+b') as indexfile:
        indexfile.seek(offset)

        # Read and Unpack Block
        data_read = indexfile.read(struct.calcsize(block_fmt_indexfile))

        unpacked_block = struct.unpack(block_fmt_indexfile, data_read)

        is_leaf_flag = True if unpacked_block[1] else False
        leaf_node_size = struct.calcsize(record_fmt_indexfile_leaf)
        inner_node_size = struct.calcsize(record_fmt_indexfile_inner)

        b: Block_Indexfile = Block_Indexfile(
                point_dim=point_dim, 
                is_leaf=unpacked_block[1], 

                size_of_record= leaf_node_size if is_leaf_flag else inner_node_size, 
                
                block_id=b_id,

                block_size=block_size, 
                size=unpacked_block[0]
            )

        # Read and Unpack Records
        for i in range(b.max_num_of_records):
            data_read = indexfile.read(
                            leaf_node_size if b.is_leaf else inner_node_size
                        )
            unpacked_record = struct.unpack(
                    record_fmt_indexfile_leaf 
                        if b.is_leaf else 
                    record_fmt_indexfile_inner, 
                    data_read
                )
            if b.is_leaf:
                b.records[i] = \
                    Record_Indexfile(
                        dim=point_dim, 
                        is_leaf=True, 
                        record_id=unpacked_record[0], 
                        datafile_record_stored=[
                            unpacked_record[1], unpacked_record[2]
                        ], 
                        vec=[
                            unpacked_record[3 + i] \
                                for i in range(point_dim)
                        ]
                    )
            else:
                b.records[i] = \
                    Record_Indexfile(
                        dim=point_dim, 
                        is_leaf=False, 
                        record_id=unpacked_record[0], 
                        datafile_record_stored=unpacked_record[1], 
                        vec=        [
                                [
                                unpacked_record[2 + 2 * i + j] 
                                for j in range(2)
                            ] 
                            for i in range(point_dim)
                        ]
                    )  
            if i < b.size:
                b.id_to_index[unpacked_record[0]] = i

        indexfile.close()

        return b
    

