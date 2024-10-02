import math
from file_management import *
from datafile_management import *
from sys import *

point_dim = 2

# TODO : might consider removing altogether dim field from
# TODO : Record_Indexfile class
class Record_Indexfile:

    def __init__(
            self, 
            dim: int, # This can be optional meaning it defaults at 2 and needs to be removed from fields 
                      # because we can use it and discard it as it contains information that can be easily extracted using len(vec)
            is_leaf: bool, 
            datafile_record_stored: list[int, int] | int, # pointer to either a block on R-tree in indexfile
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

    def __str__(self) -> str:
        output = 'r_id: ' + str(self.record_id) + ' coords:' + str(self.vec) 

        return output 
    
    def set_record_id(self, new_id):
        self.record_id = new_id
            
    def is_dominated(self, other):
        other_point: Record_Datafile = record_load_datafile(other.datafile_record_stored[0], other.datafile_record_stored[1])

        if self.is_leaf:
            self_point: Record_Datafile = record_load_datafile(self.datafile_record_stored[0], self.datafile_record_stored[1])

            for dim in range(self.dim):
                if self_point.vec[dim] < other_point[dim]:
                    return False
                
            return True
        
        else:
            for dim in range(self.dim):
                if self.vec[dim][0] < other_point[dim]:
                    return False
            
            return True

            
    # Calculates the minimum distance between the beginning of the axis
    # and the record which it is either a bounding box or a point
    # TODO: we need to chech about "negative" distances, meaning that
    # TODO: beginning of axis is inside the bb
    def calc_min_dist(self, other = None):
        if self.is_leaf :
            point: Record_Datafile = record_load_datafile(self.datafile_record_stored[0], self.datafile_record_stored[1])
            distance = math.sqrt(sum(x**2 for x in point.vec))
        else: # is a bounding box
            closest_points = []
            for i in range(self.dim): # for each dimension check which bound is closest to 0
                if 0 <= self.vec[i][0]:
                    closest_points.append(self.vec[i][0])
                elif 0 >= self.vec[i][1]:
                    closest_points.append(self.vec[i][1])
                else:
                    closest_points.append(0)
            
            distance = math.sqrt(sum(x**2 for x in closest_points))
        
        return distance
            
    def __lt__(self, other_record):
        self_min_dist = self.calc_min_dist()
        other_min_dist = other_record.calc_min_dist()
        
        return self_min_dist < other_min_dist
        

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

    def __str__(self) -> str:
        output = 'block id:' + str(self.block_id) + ' size: ' + str(self.size) + ' percentage of fullness: ' + str(self.size/self.max_num_of_records)
        

        return output

    def give_next_available_record_id(self) -> int:
        next_record_id = self.record_id_counter
        self.record_id_counter += 1
        return next_record_id

    def add_record(self, r: Record_Indexfile):

        if self.size < self.max_num_of_records:
            self.records[self.size] = r
            self.id_to_index[r.record_id] = self.size
            self.size += 1



    def calculate_MBR(self):
        lower_bound = [float_info.max for _ in range(self.point_dim)]
        upper_bound = [-float_info.max for _ in range(self.point_dim)]

        if self.is_leaf :

            for record in self.records[:self.size]:
                for dimension in range(point_dim):

                    if lower_bound[dimension] > record.vec[dimension]:
                        lower_bound[dimension] = record.vec[dimension]

                    if upper_bound[dimension] < record.vec[dimension]:
                        upper_bound[dimension] = record.vec[dimension]

        else:
            for record in self.records[:self.size]:
                for dimension in range(point_dim):

                    if lower_bound[dimension] > record.vec[dimension][0]:
                        lower_bound[dimension] = record.vec[dimension][0]

                    if upper_bound[dimension] < record.vec[dimension][1]:
                        upper_bound[dimension] = record.vec[dimension][1]

        
        return list(zip(lower_bound, upper_bound))






    def remove_record(self, r_id_for_removal: int):

        # if the record exists in the dictionary delete it and reform the records[] and the dictionary
        if r_id_for_removal in self.id_to_index:
            pos_in_records = self.id_to_index[r_id_for_removal]
            last_element = self.records[self.size - 1]
            
            # put the last element of the records array to the place of the element you want to remove and replace the last element with a dummy
            self.records[pos_in_records] = last_element
            self.records[self.size - 1] = Record_Indexfile(point_dim)
            self.size -= 1

            # Delete the record from the dictionary and update the "last" element
            self.id_to_index[last_element.record_id] = self.id_to_index.pop(r_id_for_removal)

    #! check for case for an empty root
    def calculate_bounding_box(self) -> list[list[int, int]]:
        bounding_box = []

        if self.is_leaf:

            for dim in point_dim:

                lowest = self.records[0].vec[dim] if self.size else float_info.max
                highest = self.records[0].vec[dim] if self.size else -float_info.max
                for rec_index in self.size:
                    if self.records[rec_index].vec[dim] < lowest:
                        lowest = self.records[rec_index].vec[dim]
                    elif self.records[rec_index].vec[dim] > highest:
                        highest = self.records[rec_index].vec[dim]
                bounding_box.append([lowest, highest])
        else:
            
            for dim in point_dim:

                lowest = self.records[0].vec[dim][0]
                highest = self.records[0].vec[dim][1]
                for rec_index in self.size:
                    if self.records[rec_index].vec[dim][0] < lowest:
                        lowest = self.records[rec_index].vec[dim][0]
                    if self.records[rec_index].vec[dim][1] > highest:
                        highest = self.records[rec_index].vec[dim][1]
                bounding_box.append([lowest, highest])

        return bounding_box