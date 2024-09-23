point_dim = 2

# TODO : might consider removing altogether dim field from
# TODO : Record_Indexfile class
class Record_Indexfile:

    def __init__(
            self, 
            dim: int, # This can be optional meaning it defaults at 2 and needs to be removed from fields 
                      # because we can use it and discard it as it contains information that can be easily extracted using len(vec)
            is_leaf: bool, 
            datafile_record_stored: int | list[int, int], # pointer to either a block on R-tree in indexfile
                                                          # or a unique key (block_id, record_id) to identify a record in datafile
            record_id: int = 0, 
            vec: list[float] = None
        ):
        # TODO : rename datafile_record_stored
        # datafile_record_stored will be either list or int, 
        # list when it refers to leaf node to hold block id and record id, 
        # int when Record_Indexfile refers to inner block where it will 
        # hold the block id of another Rtree node-

        self.dim = dim
        self.is_leaf = is_leaf
        self.record_id = record_id
        self.datafile_record_stored = datafile_record_stored
        if is_leaf:
            self.vec = vec if vec is not None \
                else [0.0 for _ in range(dim)]
        else:
            self.vec = vec if vec is not None \
                else [[float_info.max, -float_info.max] for _ in range(dim)] #init in the extreme values of floats
        

class Block_Indexfile:

    def __init__(
            self, 
            point_dim, 
            is_leaf: bool, 
            size_of_record, 
            block_id=0, 
            block_size=2 ** 15, 
            size=0
        ):
        self.point_dim = point_dim
        self.is_leaf = is_leaf
        self.block_size = block_size
        self.block_id = block_id
        self.max_num_of_records = block_size // size_of_record
        self.size = size
        self.records = [
                Record_Indexfile(
                    dim=point_dim, 
                    is_leaf=is_leaf,
                    datafile_record_stored=[0, 0] if is_leaf else 0
                ) 
                for _ in range(self.max_num_of_records)
            ]
        self.id_to_index: dict = dict()
        self.record_id_counter = 1

    def give_next_available_record_id(self) -> int:
        next_record_id = self.record_id_counter
        self.record_id_counter += 1
        return next_record_id

    def add_record(self, r: Record_Indexfile):

        if self.size < self.max_num_of_records:
            self.records[self.size] = r
            self.id_to_index[r.record_id] = self.size
            self.size += 1

    def remove_record(self, record_id: int):

        if record_id in self.id_to_index:
            i = self.id_to_index[record_id]
            self.records[i] = self.records[self.size - 1]
            self.records[self.size - 1] = Record_Indexfile(point_dim)
            self.size -= 1
            del self.id_to_index[record_id]