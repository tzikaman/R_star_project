import struct




point_dim = 3
block_fmt_datafile = '>II'
block_fmt_indexfile = '>II?'
record_fmt_datafile = '>II30s' + ''.join(['d' for _ in range(point_dim)])
record_fmt_indexfile_inner = '>II' + ''.join(['d' for _ in range(point_dim * point_dim)])
record_fmt_indexfile_child = '>II' + ''.join(['d' for _ in range(point_dim)])




class Record_Datafile:

    def __init__(self, dim: int, record_id: int = 0, id: int = 0, name: str = 'aaaaaaaaaaaaaaa', vec: list[float] = None):  # TODO : search all differences between byte strings and strings in python, especially when it comes to character size, encoding and transitions between the two

        self.dim = dim
        self.record_id = record_id
        self.id = id
        self.name = name
        self.vec = vec if vec is not None else [0.0 for _ in range(dim)]


class Record_Indexfile:

    def __init__(self, dim: int, is_child: bool, record_id: int = 0, datafile_record_stored: tuple[int, int] = (0, 0), vec: list[float] = None):
        self.dim = dim
        self.is_child = is_child
        self.record_id = record_id
        self.datafile_record_stored = datafile_record_stored
        if is_child:
            self.vec = vec if vec is not None else [0.0 for _ in range(dim)]
        else:
            self.vec = vec if vec is not None else [[0.0 for _ in range(dim)] for _ in range(dim)]
        



class Block:

    def __init__(self, point_dim, size_of_record, block_size=2**15, size=0):

        self.block_size = block_size  # in Bytes
        self.max_num_of_records = self.block_size // size_of_record
        self.size = size  # current number of non dummy records
        self.records = [Record_Datafile(point_dim) for _ in range(self.max_num_of_records)]
        self.id_to_index: dict = dict()

    def add_record(self, r: Record_Datafile):

        if self.size < self.max_num_of_records:
            self.records[self.size] = r
            self.id_to_index[r.record_id] = self.size
            self.size += 1

    def remove_record(self, record_id: int):

        if record_id in self.id_to_index:
            i = self.id_to_index[record_id]
            self.records[i] = self.records[self.size - 1]
            self.records[self.size - 1] = Record_Datafile(point_dim)
            self.size -= 1
            del self.id_to_index[record_id]




class Block_IndexFile:

    def __init__(self, dim, is_child: bool, size_of_record, block_size=2**15, size=0):
        self.dim = dim
        self.is_child = is_child
        self.block_size = block_size
        self.max_num_of_records = block_size // size_of_record
        self.size = size
        self.records = [Record_Indexfile(dim=dim, is_child=is_child) for _ in range(self.max_num_of_records)]

    def add_record(self, r: Record_Datafile):

        if self.size < self.max_num_of_records:
            self.records[self.size] = r
            self.size += 1

    def remove_record(self, record_id: int):

        if record_id in self.id_to_index:
            i = self.id_to_index[record_id]
            self.records[i] = self.records[self.size - 1]
            self.records[self.size - 1] = Record_Datafile(point_dim)
            self.size -= 1
            del self.id_to_index[record_id]


def write_block(b: Block, datafile, offset):

    datafile.seek(offset)

    packed_record = struct.pack(block_fmt_datafile, b.block_size, b.size)

    datafile.write(packed_record)

    for i in range(b.max_num_of_records):
        args = [b.records[i].record_id, b.records[i].id, b.records[i].name.encode('utf-8')] + b.records[i].vec
        packed_record = struct.pack(record_fmt_datafile, *args)
        datafile.write(packed_record)

    
def load_block(datafile, offset) -> Block:

    datafile.seek(offset)

    data_read = datafile.read(struct.calcsize(block_fmt_datafile))

    unpacked_block = struct.unpack(block_fmt_datafile, data_read)

    b: Block = Block(record_dim=point_dim, size_of_record=struct.calcsize(record_fmt_datafile), block_size=unpacked_block[0], size=unpacked_block[1])

    for i in range(b.max_num_of_records):
        data_read = datafile.read(struct.calcsize(record_fmt_datafile))
        unpacked_record = struct.unpack(record_fmt_datafile, data_read)
        b.records[i] = Record_Datafile(point_dim, record_id=unpacked_record[0], id=unpacked_record[1], name=unpacked_record[2].decode('utf-8').replace('\0', ''), vec=[unpacked_record[i] for i in range(3, len(unpacked_record))])
        if i < b.size:
            b.id_to_index[unpacked_record[0]] = i
    return b




if __name__ == '__main__':

    offsets = []
    b1: Block = Block(record_dim=point_dim, size_of_record=struct.calcsize(record_fmt_datafile))
    b2: Block = Block(record_dim=point_dim, size_of_record=struct.calcsize(record_fmt_datafile))
    b1.add_record(Record_Datafile(dim=point_dim, record_id=b1.size, id=1234, name='pure nigger', vec=[1.0, 2.0, 3.0]))
    b1.add_record(Record_Datafile(dim=point_dim, record_id=b1.size, id=4567, name='nigga of the purest', vec=[3.0, 4.0, 5.0]))
    b1.add_record(Record_Datafile(dim=point_dim, record_id=b1.size, id=8910, name='pertouli', vec=[5.0, 6.0, 7.0]))
    b2.add_record(Record_Datafile(dim=point_dim, record_id=b2.size, id=1000, name='the great dick of thanos', vec=[7.0, 8.0, 9.0]))
    b2.add_record(Record_Datafile(dim=point_dim, record_id=b2.size, id=2000, name='petros boglanitis', vec=[9.0, 10.0, 11.0]))

    with open('C:\\Users\\User\\Documents++\\programming\\code\\database_technologies\\Database_Technologies_Assignment\\R_star_project\\data_file_example.log', 'wb') as datafile:

        datafile.seek(0)
        offsets.append(datafile.tell())
        write_block(b1, datafile, datafile.tell())
        offsets.append(datafile.tell())
        write_block(b2, datafile, datafile.tell())
        datafile.close()

    with open('C:\\Users\\User\\Documents++\\programming\\code\\database_technologies\\Database_Technologies_Assignment\\R_star_project\\data_file_example.log', 'r+b') as datafile:
        
        b1 = load_block(datafile, offsets[1])
        b2 = load_block(datafile, offsets[0])
        datafile.close()