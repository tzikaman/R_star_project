import osmium
import numpy as np

# class GetNamesAndLocs(osmium.SimpleHandler):
    
#     def __init__(self):
#         osmium.SimpleHandler.__init__(self)
        
#         self.ids  = []
#         self.lats = []
#         self.lons = []
        
#     def node(self, n):
#         self.ids.append(n.id)
#         self.lats.append(n.location.lat)
#         self.lons.append(n.location.lon)



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





# h = GetNamesAndLocs()
# h.apply_file('C:\\Users\\User\\Documents++\\university\\'
#              'subjects\\databases_technologies\\\project\\map.osm')

# print(f"{len(h.ids)}\n{len(h.lats)}\n{len(h.lons)}")