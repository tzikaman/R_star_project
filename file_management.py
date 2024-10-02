from datafile_management import *
from indexfile_management import *



import struct



from sys import float_info
from os import path

#########################################################################

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


#! check again all double loops in list initialisations

number_of_records_in_datafile = 0

number_of_blocks_in_datafile = 0

datafile_blocks_offsets = dict()

offset_for_next_block = struct.calcsize(block_fmt_datafile)

# this is a list of datafile-block-ids that due to record deletion
# they have empty record space, so new elements are inserted there
datafile_blocks_with_bubbles = []

# holds block id of last datafile block
#! remember to change code so this always stores correct value
datafile_last_block_id = 0

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
record_fmt_indexfile = '>III' + ''.join(['d' for _ in range(2 * point_dim)])



# It writes a specified block with all its contents in the specified
# offset of the specified file and returns where the file pointer stopped 

def block_write_datafile(b: Block_Datafile, datafile_name, offset) -> int:

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

        new_offset_for_next_block: int = datafile.tell()

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
                            [b.records[i].vec[j] for j in range(point_dim)] + \
                                [0.0 for j in range(point_dim)] # dummy numbers to fill space
                packed_record = struct.pack(record_fmt_indexfile, *args)
                indexfile.write(packed_record)
            else:
                args = [b.records[i].record_id, 
                        b.records[i].datafile_record_stored, # 0 is dummy number to fill space
                        0] + \
                    [extreme_value for dimension in b.records[i].vec for extreme_value in dimension] # it is an MBR because its not a leaf
                packed_record = struct.pack(record_fmt_indexfile, *args)
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




def block_load_indexfile(indexfile_name, block_id, offset) -> Block_Indexfile:


    with open(indexfile_name, 'r+b') as indexfile:
        indexfile.seek(offset)

        # Read and Unpack Block
        data_read = indexfile.read(struct.calcsize(block_fmt_indexfile))

        unpacked_block = struct.unpack(block_fmt_indexfile, data_read)

        b: Block_Indexfile = Block_Indexfile(
                point_dim=point_dim, 
                is_leaf=unpacked_block[1], 

                size_of_record=struct.calcsize(record_fmt_indexfile),
                block_id=block_id, 

                block_size=block_size, 
                size=unpacked_block[0]
            )

        # Read and Unpack Records
        for i in range(b.max_num_of_records):
            data_read = indexfile.read(struct.calcsize(record_fmt_indexfile))
            unpacked_record = struct.unpack(record_fmt_indexfile, data_read)
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

 
"""
adds point to datafile
"""
def add_point_to_datafile(
        point_id: int,
        point_coords: list[float],
        point_name: str,
        datafile_name
):
    datafile_block_to_insert: Block_Datafile = block_load_datafile(
                                    datafile_name,
                                    offset=\
                                        datafile_blocks_offsets[datafile_blocks_with_bubbles.pop(0)]
                                        if datafile_blocks_with_bubbles else
                                        (offset_for_next_block - struct.calcsize(block_fmt_datafile))
                                )

    new_block_created: bool = False

    if datafile_block_to_insert.is_full():
        new_block_created = True
        datafile_block_to_insert = Block_Datafile(
                                        block_id=Block_Datafile.get_next_available_block_id(),
                                        point_dim=point_dim,
                                        size_of_record=struct.calcsize(record_fmt_datafile),
                                        block_size=block_size
                                    )
    
    datafile_block_to_insert.add_record(
        Record_Datafile(
            dim=point_dim,
            record_id=datafile_block_to_insert.get_next_available_record_id(),
            name=point_name,
            vec=point_coords
        )
    )

    if new_block_created:
        datafile_blocks_offsets[datafile_block_to_insert.block_id] = offset_for_next_block
        offset_for_next_block = block_write_datafile(
            datafile_block_to_insert,
            datafile_name,
            offset=offset_for_next_block
        )
        datafile_last_block_id = datafile_block_to_insert.block_id
    else:
        block_write_indexfile(
            datafile_block_to_insert,
            datafile_name,
            offset=datafile_blocks_offsets[datafile_block_to_insert.block_id]
        )


def datafile_initial_load(datafile_name):
    if path.exists(datafile_name):
        with open(datafile_name, 'rb') as datafile:
            packed_block_0_offset = datafile.read(struct.calcsize('>I'))
            block_0_offset = struct.unpack('>I', packed_block_0_offset)[0]
            datafile.seek(block_0_offset)
            data_read = struct.unpack(
                '>IIII',
                datafile.read(struct.calcsize('>IIII'))
            )
            number_of_records_in_datafile = data_read[0],
            number_of_blocks_in_datafile = data_read[1],
            offset_for_next_block = data_read[2],
            datafile_last_block_id = data_read[3]
            length_of_bubble_list = struct.unpack(
                '>I',
                datafile.read(struct.calcsize('>I'))
            )[0]
            datafile_blocks_with_bubbles = list(
                struct.unpack(
                    '>' + (length_of_bubble_list * 'I'),
                    datafile.read(struct.calcsize('>' + (length_of_bubble_list * 'I')))
                )
            )
            datafile_blocks_offsets = []
            for _ in range(number_of_blocks_in_datafile):
                datafile_blocks_offsets.append(
                    struct.unpack(
                        '>II',
                        datafile.read(struct.calcsize('>II'))
                    )
                )
            datafile_blocks_offsets = dict(datafile_blocks_offsets)
    else:
        with open(datafile_name, 'wb') as datafile:
            number_of_records_in_datafile = 0
            number_of_blocks_in_datafile = 0
            offset_for_next_block = struct.calcsize('>I')
            datafile_last_block_id = None