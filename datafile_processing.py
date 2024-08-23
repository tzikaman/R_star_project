import struct




record_dim = 2
block_fmt = '>IIII'
record_fmt = '>II30s' + ''.join(['d' for _ in range(record_dim)])




class Record:

    def __init__(self, dim: int, record_id: int = 0, id: int = 0, name: str = 'aaaaaaaaaaaaaaa', vec: list[float] = None):

        self.is_legit = False
        self.dim = dim
        self.record_id = record_id
        self.id = id
        self.name = name
        self.vec = vec if vec is not None else [0.0 for _ in range(dim)]
        

class Block:

    def __init__(self, record_dim, size_of_record, size=0, index=0):

        self.block_size = 2**15  # in Bytes
        self.max_num_of_records = self.block_size // size_of_record
        self.size = size  # current number of non dummy records
        self.records = [Record(record_dim) for _ in range(self.max_num_of_records)]
        self.index = index  # marks the index of the first record (first meaning smallest index in the array) that is a dummy
        self.id_to_index: dict = dict()

    def add_record(self, r: Record):

        if self.size < self.max_num_of_records:
            self.records[self.index] = r
            self.id_to_index[r.record_id] = self.index
            self.size += 1
            self.index += 1

    def remove_record(self, record_id: int):

        if record_id in self.id_to_index:
            i = self.id_to_index[record_id]
            self.records[i] = self.records[self.index - 1]
            self.records[self.index - 1] = Record(record_dim)
            self.index -= 1
            self.size -= 1
            del self.id_to_index[record_id]




def write_block(b: Block, datafile, offset):

    datafile.seek(offset)

    packed_record = struct.pack(block_fmt, b.block_size, b.max_num_of_records, b.size, b.index)

    datafile.write(packed_record)

    for i in range(b.max_num_of_records):
        packed_record = struct.pack(record_fmt, b.records[i].record_id, b.records[i].id, b.records[i].name.encode('utf-8'), b.records[i].vec[0], b.records[i].vec[1])
        datafile.write(packed_record)

    
def load_block(datafile, offset) -> Block:

    datafile.seek(offset)

    data_read = datafile.read(struct.calcsize(block_fmt))

    unpacked_block = struct.unpack(block_fmt, data_read)

    b: Block = Block(record_dim=record_dim, size_of_record=struct.calcsize(record_fmt), size=unpacked_block[2], index=unpacked_block[3])

    for i in range(b.max_num_of_records):
        data_read = datafile.read(struct.calcsize(record_fmt))
        unpacked_record = struct.unpack(record_fmt, data_read)
        b.records[i] = Record(record_dim, record_id=unpacked_record[0], id=unpacked_record[1], name=unpacked_record[2].decode('utf-8').replace('\0', ''), vec=[unpacked_record[3], unpacked_record[4]])
        if i < b.size:
            b.id_to_index[unpacked_record[0]] = i
    return b




if __name__ == '__main__':

    b: Block = None
    b1: Block = Block(record_dim=record_dim, size_of_record=struct.calcsize(record_fmt))
    b1.add_record(Record(dim=record_dim, record_id=b1.index, id=1234, name='pure nigger', vec=[1.0, 2.0]))
    b1.add_record(Record(dim=record_dim, record_id=b1.index, id=4567, name='nigga of the purest', vec=[3.0, 4.0]))
    b1.add_record(Record(dim=record_dim, record_id=b1.index, id=89, name='pertouli', vec=[5.0, 6.0]))

    with open('C:\\Users\\User\\Documents++\\programming\\code\\database_technologies\\Database_Technologies_Assignment\\R_star_project\\data_file_example.log', 'wb') as datafile:

        write_block(b1, datafile, 0)
        datafile.close()

    with open('C:\\Users\\User\\Documents++\\programming\\code\\database_technologies\\Database_Technologies_Assignment\\R_star_project\\data_file_example.log', 'r+b') as datafile:
        
        b = load_block(datafile, 0)
        b.remove_record(record_id=0)
        write_block(b, datafile, 0)
        b = None
        b = load_block(datafile, 0)
        write_block(b, datafile, 0)
        datafile.close()