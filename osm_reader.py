import random
import osmium
import numpy as np
from file_management import *
from datafile_management import *

class GetNamesAndLocs(osmium.SimpleHandler):
    
    def __init__(self):
        super().__init__()
        self.records = []  # Collect records in a list
        
    def node(self, n):
        # Append a tuple (id, lat, lon) to the list
        
        self.records.append((n.id, n.location.lat, n.location.lon))

    def get_data(self):
        # Convert the list of records to a NumPy array at the end
        return np.array(self.records, dtype=object)
    
class BatchHandler(osmium.SimpleHandler):
    def __init__(self, datafile_name, batch_size):
        osmium.SimpleHandler.__init__(self)
        self.current_batch = []
        self.current_block: Block_Datafile
        
        self.datafile_name = datafile_name
        self.batch_size = batch_size

        self.block_offsets = []
        self.offset_for_next_block = 0

        random.seed(1)

    # It automatically runs a loop and fetches a node from
    # the osm file in "n" variable
    def node(self, n):
        self.current_batch.append([n.id, n.location.lat, n.location.lon])

        if len(self.current_batch) >= self.batch_size:
            self.process_batch()


    def process_batch(self):
        if not self.current_batch:
            return
        
        self.preprocess_batch()

        self.current_block = Block_Datafile(Block_Datafile.get_next_available_block_id(),
                                               len(self.current_batch) - 2,
                                               struct.calcsize(record_fmt_datafile))
        self.block_offsets.append(self.offset_for_next_block)

        for element in self.current_batch :
            self.insert_point(element)

        self.offset_for_next_block = block_write_datafile(self.current_block,
                                                          self.datafile_name,
                                                          self.offset_for_next_block)
        
        
        self.current_batch = []



    # Adds a column with names
    def preprocess_batch(self):
        
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
        
        for sublist in self.current_batch :

            sublist.append( random_names[random.randint(0, len(random_names) - 1)] )

    def finalize(self) -> list[int]:
        if self.current_batch :
            self.process_batch()
        
        return self.block_offsets



    def insert_point(self,data):
        self.current_block.add_record(
                Record_Datafile(
                    point_dim,
                    self.current_block.get_next_available_record_id(),
                    data[0],
                    data[3],
                    [data[1], data[2]]
                )
            )


