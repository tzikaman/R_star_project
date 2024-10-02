import file_management

"""
performs serial search to find a specific point in the datafile
"""
def serial_search(
        point_id: int,
        datafile_name
) -> file_management.Record_Datafile:
    for block_offset in file_management.datafile_blocks_offsets.values():
        datablock = file_management.block_load_datafile(
                        datafile_name,
                        offset=block_offset
                    )
        for entry_index in range(datablock.size):
            if datablock.records[entry_index].id == point_id:
                # found the record, therefore returns it
                return datablock.records[entry_index]
    return None # record is not in the datafile