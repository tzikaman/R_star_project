point_dim = 2

# TODO : might consider removing altogether dim field from
# TODO : Record_Indexfile class
class Record_Datafile:

    def __init__(
            self, 
            dim: int, #the dimension of the data points
            record_id: int = 0, # the unique identifier of the record in a block
            id: int = 0, # the unique identifier of the point (the only unique id that differentiates the point from every other point)
            name: str = 'aaaaaaaaaaaaaaa', #the name of the point
            vec: list[float] = None #the coordinates of the given point
        ):  

        # TODO : search all differences between byte strings 
        # TODO : and strings in python, especially when it 
        # TODO : comes to character size, encoding and 
        # TODO : transitions between the two

        self.dim = dim
        self.record_id = record_id
        self.id = id
        self.name = name
        self.vec = vec if vec is not None else [0.0 for _ in range(dim)] # init list vec with the given coords else init a zero list

class Block_Datafile:

    block_id_counter = 0 # It is a shared variable through all instances
    def __init__(
            self, 
            block_id,
            point_dim, 
            size_of_record, # ? Cant we extract that information given the line 35 ?
            block_size=2**15, #maximum size that a block can reach
            size=0, #current size of the block , maybe rename that to a counter
            record_id_counter=0 # what's the difference between that and size ?
        ):
        self.block_id = block_id
        self.point_dim = point_dim
        self.block_size = block_size  # in Bytes
        self.max_num_of_records = self.block_size // size_of_record
        self.size = size  # current number of non dummy records
        self.records = [Record_Datafile(point_dim) for _ in range(self.max_num_of_records)] # ? cant we find a way to save 1 each time because
                                                                                            # that means that from the beginning we will occupy all
                                                                                            # of the space we ever gonna need in a block thus having an overhead in the initial steps? 
                                                                                            # essentially this line preoccupies the max records that a block is gonna need
        self.id_to_index: dict = dict()
        self.record_id_counter = record_id_counter

    def __str__(self):
        return"BLOCK ID: " + str(self.block_id) + \
              " block_size: " + str(self.block_size) + \
              " max_records: " + str(self.max_num_of_records)  + \
              " size: " + str(self.size) + str([r.record_id for r in self.records])

    def is_full(self):
        return self.size == self.max_num_of_records

    def add_record(self, r: Record_Datafile):

        # This will be executed every time because each time we check if the block is full but it's good to keep it to prevent crashes
        if self.size < self.max_num_of_records: 
            self.records[self.size] = r # this is allowed because the list "records[]" is full of dummy records
            self.id_to_index[r.record_id] = self.size  # the new record will have an index in "records[]" related to the order in which is inserted
            self.size += 1
            return r.record_id

    def remove_record(self, r_id_for_removal: int):

        # if the record exists in the dictionary delete it and reform the records[] and the dictionary
        if r_id_for_removal in self.id_to_index:
            pos_in_records = self.id_to_index[r_id_for_removal]

            last_element = self.records[self.size - 1]
            # put the last element of the records array to the place of the element you want to remove and replace the last element with a dummy
            self.records[pos_in_records] = last_element
            self.records[self.size - 1] = Record_Datafile(point_dim)
            self.size -= 1

            # Delete the record from the dictionary and update the "last" element
            self.id_to_index[last_element.record_id] = self.id_to_index.pop(r_id_for_removal)            
            

    def get_next_available_record_id(self):
        next_record_id = self.record_id_counter
        self.record_id_counter += 1
        return next_record_id

    @staticmethod # it's static because you can access it even if you dont have any instances of the of the object 
                  # in the beggining it will give you 0 to create your first object
    def get_next_available_block_id():
        next_block_id = Block_Datafile.block_id_counter
        Block_Datafile.block_id_counter += 1
        return next_block_id 