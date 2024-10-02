import osm_reader
import struct

block_fmt_datafile = '>IIII'
block_fmt_indexfile = '>I?'
point_dim = 2

record_fmt_datafile = '>IQ' + ''.join(['d' for _ in range(point_dim)])
record_fmt_indexfile_inner = '>II' + ''.join(['d' for _ in range(2 * point_dim)])
record_fmt_indexfile_leaf = '>III' + ''.join(['d' for _ in range(point_dim)])

osm_path = "C:\\Users\\Public\\Documents\\map.osm"

print(struct.calcsize(record_fmt_datafile))
print(struct.calcsize(block_fmt_datafile))