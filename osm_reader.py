import osmium

class GetNamesAndLocs(osmium.SimpleHandler):
    
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.ids  = []
        self.lats = []
        self.lons = []
        
    def node(self, n):
        self.ids.append(n.id)
        self.lats.append(n.location.lat)
        self.lons.append(n.location.lon)




# h = GetNamesAndLocs()
# h.apply_file('C:\\Users\\User\\Documents++\\university\\'
#              'subjects\\databases_technologies\\\project\\map.osm')

# print(f"{len(h.ids)}\n{len(h.lats)}\n{len(h.lons)}")