import sys
from file_management import * 
from datafile_management import *

import math
import copy
import heapq

from indexfile_management import *

point_dim = 2

class Rtree:

    def __init__(self, 
            index_file_name,
            root_block_offset = None,
            forced_reinsert_enable = True,
            maximum_num_of_records = None,
            minimum_num_of_records = None,
            num_of_blocks = 1,
            num_of_leaves = 1,
            height = 1,
            root_bounding_box = None,
            block_id_to_file_offset = dict(),
            block_id_index_counter = 1,
            offset_for_next_block_to_enter = \
                struct.calcsize('>I')
        ):
        self.index_file_name = index_file_name
        self.forced_reinsert_enable = forced_reinsert_enable
        self.num_of_blocks = num_of_blocks
        self.num_of_leaves = num_of_leaves
        self.height = height

        #init the bb of the root but if not given put some placeholder values
        if root_bounding_box is None:
            self.root_bounding_box = \
                [[float_info.max, -float_info.max] for _ in range(point_dim)]
        else:
            self.root_bounding_box = root_bounding_box

        # TODO : Check if can gauge height of Tree 
        # TODO : based on number of nodes it has

        self.block_id_to_file_offset = \
            block_id_to_file_offset
        
        # TODO : Consider depending on how the block 
        # TODO : id would work if the dict could be changed to a list
        self.block_id_index_counter = \
            block_id_index_counter
        
        # offset is the place in indexfile from where
        # the rtree structure will get written
        self.offset_for_next_block_to_enter = \
            offset_for_next_block_to_enter
        
        if root_block_offset is None:
            # in this case it is indicated that 
            # an empty Rtree is created, so the 
            # indexfile will be created now 
            with open(self.index_file_name, 'wb') as indexfile:
                indexfile.write(
                    struct.pack(
                        '>I',
                        0  # at this stage it is irrelevant what the first 
                           # unsigned integer will be, thus the value 0. The
                           # first time this first unsigned integer in the file
                           # will hold value is when the whole Rtree is stored
                           # for the first time in the file, during program 
                           # exit
                    )
                )
                indexfile.close()
            '''
            # create first block
            self.root: Block_Indexfile = \
                Block_Indexfile(
                    point_dim=point_dim, 
                    is_leaf=True, 
                    size_of_record=\
                        struct.calcsize(record_fmt_indexfile_leaf), 
                    block_id=0
                )

            # write first entry for the dictionary
            self.block_id_to_file_offset[self.root.block_id] = \
                self.offset_for_next_block_to_enter
            
            # write first block to indexfile
            self.offset_for_next_block_to_enter = \
                block_write_indexfile(
                    self.root,
                    self.index_file_name,
                    offset=self.offset_for_next_block_to_enter
                )
            '''
        else:
            self.root = \
            block_load_indexfile(self.index_file_name,
                                 root_block_offset)
            
            
        # The M parameter is the maximum number of records a block can hold
        # unless specified otherwise
        if maximum_num_of_records is None:
            self.maximum_num_of_records = \
                self.root.max_num_of_records
        else:
            self.maximum_num_of_records = \
                maximum_num_of_records
            
        # The m parameter 40% of the max number of records -> but i think it should be 50% 
        if minimum_num_of_records is None:
            self.minimum_num_of_records = \
                math.floor(0.4 * self.maximum_num_of_records)
        else:
            self.minimum_num_of_records = \
                minimum_num_of_records    


    def _give_next_available_block_id(self) -> int:
        next_id = self.block_id_index_counter
        self.block_id_index_counter += 1
        return next_id

    #! check that thing about forced reinsert and 
    #! Rtree gaining height in the process, whether
    #! entries will remember at what level they need
    #! to enter

    # TODO : check if it is safe to check target_level on an entry to be 
    # TODO : inserted to the Rtree and compare it with Rtree height to 
    # TODO : check whether current entry needs to enter in a leaf node 
    # TODO : or in an inner node
    def _calculate_bb_size_increase(
            self, 
            entry_is_leaf: bool, 
            bounding_box, 
            inserted_point
        ):
        new_bb = []
        # For each dimension
        for i in range(point_dim):
            # TODO : consider rewriting below 'if block' as is 'else block'
            if entry_is_leaf:
                if inserted_point[i] >= bounding_box[i][0] \
                          and inserted_point[i] <= bounding_box[i][1]:
                    new_bb.append((bounding_box[i][0], bounding_box[i][1]))
                elif inserted_point[i] < bounding_box[i][0]:
                    new_bb.append((inserted_point[i], bounding_box[i][1]))
                else:
                    new_bb.append((bounding_box[i][0], inserted_point[i]))

            else: # I dont understand what the diff will be if the entry is leaf or not
                  # and also I dont think inserted_point[i][j] is a valid command
                new_bb.append(
                    (
                        inserted_point[i][0] 
                        if inserted_point[i][0] < bounding_box[i][0] 
                        else bounding_box[i][0],
                        inserted_point[i][1] 
                        if inserted_point[i][1] > bounding_box[i][1]
                        else bounding_box[i][1]
                    )
                )
                
        area_former_bb = 1
        for element in bounding_box:
            area_former_bb *= (element[1] - element[0])

        area_new_bb = 1
        for element in new_bb:
            area_new_bb *= (element[1] - element[0])

        return (area_new_bb - area_former_bb)

    def _forced_reinsert(
            self, 
            current_node: Block_Indexfile, 
            bb, 
            overflow_element_datafile_record_stored, 
            overflow_element_coords
        ):
        block_bb_center = [(elem[1] + elem[0]) / 2 for elem in bb]
        entry_with_distance = []
        # add overflow_element to the calculation
        dist_from_bb_center = 0.0
        for j in range(point_dim):
            if current_node.is_leaf:
                dist_from_bb_center += (block_bb_center[j] - \
                                    overflow_element_coords[j]) ** 2
            else:
                dist_from_bb_center += \
                (block_bb_center[j] - 
                 ((overflow_element_coords[j][0] + \
                   overflow_element_coords[j][1]) / 2)) ** 2
        dist_from_bb_center = math.sqrt(dist_from_bb_center)
        entry_with_distance.append((-1, dist_from_bb_center))
        # here there is the convention that the overflow element will have
        # -1 for identifier as there is no record of it in the block so there 
        # is no record id

        for i in range(0, current_node.size):
            dist_from_bb_center = 0.0
            for j in range(point_dim):
                if current_node.is_leaf:
                    dist_from_bb_center += (block_bb_center[j] - 
                                current_node.records[i].vec[j]) ** 2
                else:
                    dist_from_bb_center += \
                        (block_bb_center[j] - \
                         ((current_node.records[i].vec[j][0] + \
                           current_node.records[i].vec[j][1]) / 2)) ** 2
            dist_from_bb_center = math.sqrt(dist_from_bb_center)
            entry_with_distance.append(
                (current_node.records[i].record_id, dist_from_bb_center))
        entry_with_distance.sort(key=lambda x : x[1])
        entry_with_distance.reverse()

        elements_for_reinsertion = []
        overflow_element_selected_for_reinsertion: bool = False

        for i in range(current_node.size // 3):
            if entry_with_distance[i][0] == -1:
                elements_for_reinsertion.append(
                    (
                        current_node.is_leaf,
                        overflow_element_datafile_record_stored,
                        overflow_element_coords
                    )
                )
                overflow_element_selected_for_reinsertion = True
            else:
                pos = current_node.id_to_index[entry_with_distance[i][0]]
                current_node.id_to_index[current_node.records[current_node.size - 1]
                                         .record_id] = pos
                elements_for_reinsertion.append(
                        (
                            current_node.is_leaf, 
                            current_node.records[pos].datafile_record_stored.copy(),
                            copy.deepcopy(current_node.records[pos].vec)
                        )
                    )
                # del current_node.records[pos] #!
                current_node.records[pos] = \
                    current_node.records[current_node.size - 1]                
                current_node.records[current_node.size - 1] = \
                    Record_Indexfile(
                        dim=point_dim, 
                        is_leaf=current_node.is_leaf, 
                        datafile_record_stored= [0, 0]
                            if current_node.is_leaf else 0,
                        record_id=current_node
                        .give_next_available_record_id()
                    )
                current_node.size -= 1
        
        if not overflow_element_selected_for_reinsertion:
            current_node.add_record(
                Record_Indexfile(
                    point_dim,
                    is_leaf=current_node.is_leaf,
                    datafile_record_stored=\
                        overflow_element_datafile_record_stored,
                    record_id=current_node
                        .give_next_available_record_id(),
                    vec=overflow_element_coords
                )
            )

        # write updated block to storage
        block_write_indexfile(
            current_node,
            self.index_file_name,
            self.block_id_to_file_offset[current_node.block_id]
        )
        
        # calculate new bounding box for node
        bb = []
        for i in range(point_dim):
            bb.append(
                [current_node.records[0].vec[i]
                 if current_node.is_leaf else
                 current_node.records[0].vec[i][0],
                 current_node.records[0].vec[i]
                 if current_node.is_leaf else
                 current_node.records[0].vec[i][1]]
            )
            for j in range(1, current_node.size):
                if current_node.is_leaf:
                    if current_node.records[j].vec[i] < bb[i][0]:
                        bb[i][0] = current_node.records[j].vec[i]
                    elif current_node.records[j].vec[i] > bb[i][1]:
                        bb[i][1] = current_node.records[j].vec[i]
                else:
                    if current_node.records[j].vec[i][0] < bb[i][0]:
                        bb[i][0] = current_node.records[j].vec[i][0]
                    if current_node.records[j].vec[i][1] > bb[i][1]:
                        bb[i][1] = current_node.records[j].vec[i][1]

        # reinsertion cannot start here because bb of 
        # current node change must be first updated to 
        # the node above
        return (bb, elements_for_reinsertion)
            
    # split node algorithm
    #! TODO : when the record containing the new element to add to
    #! TODO : the Rtree is about to enter its final block, the record
    #! TODO : id must be reassigned a valid record id
    def _split_node(
            self, 
            node_to_split: Block_Indexfile, 
            node_reference, 
            inserted_coords
        ):
        records = node_to_split.records.copy()
        records.append(
            Record_Indexfile(
                dim=point_dim, 
                is_leaf=node_to_split.is_leaf,
                datafile_record_stored=node_reference, 
                record_id=-1, 
                vec=inserted_coords
            )
        )
        best_axis_to_split = None
        bb_values_for_best_axis = None
        bb_values_for_different_distros = []
        for i in range(point_dim):
            margin_score_for_axis = 0.0
            for l in range(2):
                if node_to_split.is_leaf and l:
                    break
                if node_to_split.is_leaf:
                    records.sort(key= lambda r : r.vec[i])
                else:
                    records.sort(key= lambda r : r.vec[i][l])
                for k in range(1, self.maximum_num_of_records - 2 * 
                               self.minimum_num_of_records + 3):
                    group_1 = records[ : (self.minimum_num_of_records - 1) + k]
                    group_2 = records[(self.minimum_num_of_records - 1) + k : ]
                    bb_for_group1 = []
                    for j in range(point_dim):
                        if node_to_split.is_leaf:
                            min_for_dim = group_1[0].vec[j]
                            max_for_dim = group_1[0].vec[j]
                        else:
                            min_for_dim = group_1[0].vec[j][0]
                            max_for_dim = group_1[0].vec[j][1]
                        
                        for p in range(1, len(group_1)):
                            if node_to_split.is_leaf:
                                if group_1[p].vec[j] < min_for_dim:
                                    min_for_dim = group_1[p].vec[j]
                                elif group_1[p].vec[j] > max_for_dim:
                                    max_for_dim = group_1[p].vec[j]                            
                            else:
                                if group_1[p].vec[j][0] < min_for_dim:
                                    min_for_dim = group_1[p].vec[j][0]
                                if group_1[p].vec[j][1] > max_for_dim:
                                    max_for_dim = group_1[p].vec[j][1]
                        bb_for_group1.append([min_for_dim, max_for_dim])
                    bb_for_group2 = []
                    for j in range(point_dim):
                        if node_to_split.is_leaf:
                            min_for_dim = group_2[0].vec[j]
                            max_for_dim = group_2[0].vec[j]
                        else:
                            min_for_dim = group_2[0].vec[j][0]
                            max_for_dim = group_2[0].vec[j][1]
                        
                        for p in range(1, len(group_2)):
                            if node_to_split.is_leaf:
                                if group_2[p].vec[j] < min_for_dim:
                                    min_for_dim = group_2[p].vec[j]
                                elif group_2[p].vec[j] > max_for_dim:
                                    max_for_dim = group_2[p].vec[j]                            
                            else:
                                if group_2[p].vec[j][0] < min_for_dim:
                                    min_for_dim = group_2[p].vec[j][0]
                                if group_2[p].vec[j][1] > max_for_dim:
                                    max_for_dim = group_2[p].vec[j][1]
                        bb_for_group2.append([min_for_dim, max_for_dim])
                    bb_values_for_different_distros.append(
                        [bb_for_group1, bb_for_group2]
                    )
                    for entry in bb_for_group1:
                        margin_score_for_axis += entry[1] - entry[0]
                    for entry in bb_for_group2:
                        margin_score_for_axis += entry[1] - entry[0]
            if best_axis_to_split is None:
                best_axis_to_split = (0, margin_score_for_axis)
                bb_values_for_best_axis = bb_values_for_different_distros
            elif margin_score_for_axis < best_axis_to_split[1]:
                best_axis_to_split = (i, margin_score_for_axis)
                bb_values_for_best_axis = bb_values_for_different_distros
        # best split axis is found
        # finding best distro split
        minimum_overlap = None
        for e in range(len(bb_values_for_best_axis)):
            # find bb overlap between the two groups
            current_overlap = 1.0
            for k in range(point_dim):
                # check if there is overlap
                if bb_values_for_best_axis[e][0][k][0] >= \
                    bb_values_for_best_axis[e][1][k][1] \
                    or bb_values_for_best_axis[e][1][k][0] >= \
                        bb_values_for_best_axis[e][0][k][1]:
                    # no overlap
                    current_overlap = 0.0
                    break
                else:
                    # gradually calculate overlapping hypervolume
                    current_overlap *= (bb_values_for_best_axis[e][0][k][1] if 
                        bb_values_for_best_axis[e][0][k][1] < 
                        bb_values_for_best_axis[e][1][k][1] 
                        else bb_values_for_best_axis[e][1][k][1]) - \
                        (bb_values_for_best_axis[e][0][k][0] 
                        if bb_values_for_best_axis[e][0][k][0] > \
                        bb_values_for_best_axis[e][1][k][0] 
                        else bb_values_for_best_axis[e][1][k][0])
            # check if newly calculated overlap volume is smaller
            if minimum_overlap is None:
                minimum_overlap = (e, current_overlap)
            elif current_overlap < minimum_overlap[1]:
                minimum_overlap = (e, current_overlap)
            elif current_overlap == minimum_overlap:
                pass  # TODO : compare areas and choose smallest
        # spliting the records of node into the groups
        if node_to_split.is_leaf:
            records.sort(key= lambda r : 
                r.vec[best_axis_to_split[0]])
        else:
            records.sort(key= lambda r : 
                r.vec[best_axis_to_split[0]]
                [int(e >= (self.maximum_num_of_records - 
                        2 * self.minimum_num_of_records + 2))])
        k = (minimum_overlap[0] + 1) if minimum_overlap[0] < \
        (self.maximum_num_of_records - 2 * self.minimum_num_of_records + 2) \
        else (minimum_overlap[0] - 
            (self.maximum_num_of_records - 2 * self.minimum_num_of_records + 1))
        group_1 = records[ : (self.minimum_num_of_records - 1) + k]
        group_2 = records[(self.minimum_num_of_records - 1) + k : ]
        # creating first new block 
        # (updating old block)
        node_to_split.size = 0
        node_to_split.record_id_counter = 1
        node_to_split.id_to_index.clear()
        for i in range(node_to_split.max_num_of_records):
            if i < len(group_1):
                node_to_split.add_record(
                    Record_Indexfile(
                        point_dim,
                        node_to_split.is_leaf,
                        group_1[i].datafile_record_stored,
                        node_to_split.give_next_available_record_id(),
                        group_1[i].vec
                    )
                )
            else:
                node_to_split.records[i] = \
                    Record_Indexfile(
                        point_dim, 
                        node_to_split.is_leaf, 
                        [0, 0] if node_to_split.is_leaf else 0
                    )
        # creating second new block
        # (creating a new block)
        new_node: Block_Indexfile = \
            Block_Indexfile(
                point_dim, 
                node_to_split.is_leaf, 
                struct.calcsize(record_fmt_indexfile_leaf)
                    if node_to_split.is_leaf else
                    struct.calcsize(record_fmt_indexfile_inner), 
                self._give_next_available_block_id()
            )
        for i in range(len(group_2)):
            new_node.add_record(
                Record_Indexfile(
                    point_dim,
                    node_to_split.is_leaf,
                    group_2[i].datafile_record_stored,
                    new_node.give_next_available_record_id(),
                    group_2[i].vec
                )
            )
        self.block_id_to_file_offset[new_node.block_id] = \
            self.offset_for_next_block_to_enter
        # write new blocks to storage
        block_write_indexfile(node_to_split, 
                self.index_file_name, 
                offset=self.block_id_to_file_offset[node_to_split.block_id]
            )
        self.offset_for_next_block_to_enter = \
            block_write_indexfile(new_node, 
                        self.index_file_name, 
                        offset=self.offset_for_next_block_to_enter
                    )
        # return the block ids and the respectives bb
        return (
            (node_to_split.block_id, bb_values_for_best_axis[minimum_overlap[0]][0]), 
            (new_node.block_id, bb_values_for_best_axis[minimum_overlap[0]][1])
            )

    # TODO : See for proper memory management, see when you 
    # TODO : should pass references and when you should deep copy
    def _insert_point_recurse(
            self, 
            current_node: Block_Indexfile, 
            level, 
            bb, 
            node_reference, 
            inserted_coords, 
            target_level
        ):

        if level==target_level:
            if current_node.size == current_node.max_num_of_records:
                if level == (self.height - 2) and self.forced_reinsert_enable:
                    self.forced_reinsert_enable = False
                    result = \
                        self._forced_reinsert(
                            current_node,
                            bb,
                            node_reference,
                            inserted_coords
                        )
                    return result
                else:
                    # split algorithm for current node, might cause a split upwards
                    if current_node.is_leaf:
                        self.num_of_leaves += 1
                    new_blocks_with_bbs = \
                    self._split_node(
                        current_node, 
                        node_reference, 
                        inserted_coords
                    )
                    self.num_of_blocks += 1
                    return new_blocks_with_bbs
            else:
                # add entry to current block, no further split upwards can occur
                current_node.add_record(
                    Record_Indexfile(
                        point_dim,
                        is_leaf=current_node.is_leaf,
                        datafile_record_stored=node_reference,
                        record_id=current_node.give_next_available_record_id(),
                        vec=inserted_coords
                    )
                )

                # write changed node to storage
                block_write_indexfile(
                    current_node,
                    self.index_file_name,
                    self.block_id_to_file_offset[
                        current_node.block_id
                    ]
                )

                # check if bounding box changes
                changed: bool = False
                for i in range(point_dim):
                    if current_node.is_leaf:
                        if inserted_coords[i] < bb[i][0]:
                            bb[i][0] = inserted_coords[i]
                            changed = True
                        if inserted_coords[i] > bb[i][1]:
                            bb[i][1] = inserted_coords[i]
                            changed = True
                    else:
                        if inserted_coords[i][0] < bb[i][0]:
                            bb[i][0] = inserted_coords[i][0]
                            changed = True
                        if inserted_coords[i][1] > bb[i][1]:
                            bb[i][1] = inserted_coords[i][1]
                            changed = True
                if changed:
                    return bb
                else:
                    return None

        minimum_increased_entry: tuple[int, float] | None = None

        for i in range(current_node.size):
            current_entry_size_increase = \
                self._calculate_bb_size_increase(
                    not target_level, 
                    current_node.records[i].vec, 
                    inserted_coords
                )
            if minimum_increased_entry is None:
                minimum_increased_entry = (i, current_entry_size_increase)
            else:
                if current_entry_size_increase < minimum_increased_entry[1]:
                    minimum_increased_entry = (i, current_entry_size_increase)
                elif current_entry_size_increase == minimum_increased_entry[1]:
                    pass # TODO : calculate smallest rectangle in case increase is the same

        new_block_fetched: Block_Indexfile = \
            block_load_indexfile(
                self.index_file_name, 
                self.block_id_to_file_offset[
                    current_node.records[minimum_increased_entry[0]]
                    .datafile_record_stored
                ]
            )

        blocks_update_with_bbs = \
            self._insert_point_recurse(
                new_block_fetched, 
                level - 1, 
                current_node.records[minimum_increased_entry[0]].vec, 
                node_reference, 
                inserted_coords, 
                target_level
            )
        if blocks_update_with_bbs is None:
            # bb of node below didn't change, so no changes
            # are propagated upwards
            # no change happened to this node
            # so no need to write it to storage
            return None
        elif isinstance(blocks_update_with_bbs, tuple):
            # either a forced reinsert is about to occur
            # or a split has occured and changes must be 
            # propagated, the first element of the tuple
            # will clarify which case it is
            if isinstance(blocks_update_with_bbs[0], list):
                
                # it is a case of a forced reinsert
                # first the entry containing the bb
                # of the updated node below must be 
                # also updated to reflect the change
                current_node.records[minimum_increased_entry[0]].vec = \
                    blocks_update_with_bbs[0]
                
                # write current block in memory so 
                # change is saved and accessible 
                # during the forced reinsertion
                block_write_indexfile(
                    current_node,
                    self.index_file_name,
                    offset=self.block_id_to_file_offset[
                        current_node.block_id
                    ]
                )

                # performing an insertion for each 
                # selected element
                #! _insert_point doesn't have is_leaf parameter
                for element_to_reinsert in blocks_update_with_bbs[1]:
                    self._insert_point(
                        node_reference=element_to_reinsert[1], 
                        inserted_coords=element_to_reinsert[2], 
                        target_level=level - 1
                    )
                    
                # ending all previous recursive calls
                # regarding the insertion of current 
                # element
                return None
            else:
                # a split was performed to the node below, so 
                # changes are delivered to current node
                
                # store one of the two new blocks the entry that 
                # held the old node that got splitted
                current_node.records[minimum_increased_entry[0]].vec = \
                    blocks_update_with_bbs[0][1]

                # if current record is full and cannot store any new 
                # records (one is needed because a split has occured)
                # either a forced insert or a split must occur
                if current_node.size == current_node.max_num_of_records:
                    if self.forced_reinsert_enable:
                        self.forced_reinsert_enable = False
                        self._forced_reinsert(
                            current_node, 
                            bb, 
                            blocks_update_with_bbs[1][0], 
                            blocks_update_with_bbs[1][1]
                        )
                    else:
                        if current_node.is_leaf:
                            self.num_of_leaves += 1
                        new_blocks_with_bbs = \
                        self._split_node(
                            current_node, 
                            blocks_update_with_bbs[1][0],
                            blocks_update_with_bbs[1][1]
                        )
                        self.num_of_blocks += 1
                        return new_blocks_with_bbs
                else:
                    # insert the new entry to current node
                    current_node.add_record(
                        Record_Indexfile(
                            point_dim,
                            is_leaf=current_node.is_leaf,
                            datafile_record_stored=blocks_update_with_bbs[1][0],
                            record_id=current_node.give_next_available_record_id(),
                            vec=blocks_update_with_bbs[1][1]
                        )
                    )

                    # write changed node to storage
                    block_write_indexfile(
                        current_node,
                        self.index_file_name,
                        self.block_id_to_file_offset[
                            current_node.block_id
                        ]
                    )

                    # check if bounding box changes
                    changed: bool = False
                    for i in range(point_dim):
                        if blocks_update_with_bbs[0][1][i][0] < bb[i][0]:
                            bb[i][0] = blocks_update_with_bbs[0][1][i][0]
                            changed = True
                        if blocks_update_with_bbs[0][1][i][1] > bb[i][1]:
                            bb[i][1] = blocks_update_with_bbs[0][1][i][1]
                            changed = True
                        if blocks_update_with_bbs[1][1][i][0] < bb[i][0]:
                            bb[i][0] = blocks_update_with_bbs[1][1][i][0]
                            changed = True
                        if blocks_update_with_bbs[1][1][i][1] > bb[i][1]:
                            bb[i][1] = blocks_update_with_bbs[1][1][i][1]
                            changed = True
                    if changed:
                        return bb
                    else:
                        return None
        else:
            # the bb of new_block_fetched (the node below) changed, 
            # so this change must be reflected to the respective 
            # entry of current node
            current_node.records[minimum_increased_entry[0]] \
                .vec=blocks_update_with_bbs
            
            # write current block to storage since
            # change occured
            block_write_indexfile(
                current_node,
                self.index_file_name,
                offset=self.block_id_to_file_offset[
                    current_node.block_id
                ]
            )
            
            # check if bounding box of current_node changes
            # due to change made to bounding box of entry
            changed: bool = False
            for i in range(point_dim):
                if blocks_update_with_bbs[i][0] < bb[i][0]:
                    bb[i][0] = blocks_update_with_bbs[i][0]
                    changed = True
                if blocks_update_with_bbs[i][1] > bb[i][1]:
                    bb[i][1] = blocks_update_with_bbs[i][1]
                    changed = True
            if changed:
                return bb
            else:
                return None

    def _insert_point(self, node_reference, inserted_coords, target_level=0):
        """
        Inserts an element into the R-tree.

        :param is_leaf: 
            True if the inserted element is a bounding box or a point.
        :type is_leaf: bool
        :param inserted_point: 
            An int or a tuple[int, int] that contains a reference to the block below or to a datafile record block (block ID and record ID).
        :type inserted_point: int or tuple[int, int]
        :param inserted_coords: 
            The point to be inserted or the coordinates of a bounding box.
        :type inserted_coords: tuple[float, float] or tuple[tuple[float, float], tuple[float, float]]
        :param target_level: 
            The level in the R-tree where the insertion will happen. The default value is the leaves level.
        :type target_level: int, optional
        """
        result = \
            self._insert_point_recurse(
                self.root, 
                self.height - 1, 
                self.root_bounding_box,
                node_reference,
                inserted_coords, 
                target_level
            )

        if result is None:
            # no futher change needs to happen
            pass
        elif isinstance(result, tuple):
            # the root got split, so self.root must be updated as well
            self.root = Block_Indexfile(
                point_dim,
                is_leaf=False,
                size_of_record= \
                    struct.calcsize(record_fmt_indexfile_inner),
                block_id=self._give_next_available_block_id()
            )
            self.root.add_record(
                Record_Indexfile(
                    point_dim,
                    is_leaf=False,
                    datafile_record_stored=result[0][0],
                    record_id=self.root.give_next_available_record_id(),
                    vec=result[0][1]
                )
            )
            self.root.add_record(
                Record_Indexfile(
                    point_dim,
                    is_leaf=False,
                    datafile_record_stored=result[1][0],
                    record_id=self.root.give_next_available_record_id(),
                    vec=result[1][1]
                )
            )

            # update dictionary
            self.block_id_to_file_offset[self.root.block_id] = \
                self.offset_for_next_block_to_enter

            # write new block to storage
            self.offset_for_next_block_to_enter = \
                block_write_indexfile(
                    self.root,
                    self.index_file_name,
                    offset=self.offset_for_next_block_to_enter
                )

            # finding new root's bb
            self.root_bounding_box = []
            for i in range(point_dim):
                self.root_bounding_box.append(
                    [result[0][1][i][0] 
                     if result[0][1][i][0] < result[1][1][i][0]
                     else result[1][1][i][0],
                     result[0][1][i][1]
                     if result[0][1][i][1] > result[1][1][i][1]
                     else result[1][1][i][1]]
                )
            self.forced_reinsert_enable = True
            self.num_of_blocks += 1
            self.height += 1
        else:
            # bounding box for root changed
            self.root_bounding_box = result

    def insert_point(self, node_reference, inserted_coords):
        self._insert_point(node_reference, inserted_coords)

    # method checks if @param rectangle contains @param point
    # @param rectangle is a list of point_dim elements that are
    # lists of two elements
    # @param point is a list of point_dim elements that are the 
    # coordinates of the point
    def _rectangle_contains_point(
            point: list,
            rectangle: list
        ) -> bool:

        for i_dim in range(point_dim):

            # If outside the box at one dimension then this means
            # the rectangle does not contain the given point
            if point[i_dim] < rectangle[i_dim][0] or \
            point[i_dim] > rectangle[i_dim][1]:
                return False
            
        return True

    # method receives 2 rectangles in @global_var point_dim dimension
    # and checks whether they are intersecting each other
    def _rectangles_intersect(
            rectangle1: list,
            rectangle2: list
        ) -> bool:
        
        for i_dim in range(point_dim):

            if rectangle1[i_dim][1] < rectangle2[i_dim][0] or \
            rectangle1[i_dim][0] > rectangle2[i_dim][1]:
                return False
            
        return True
   
    def range_query(
            self,
            area: list
        ) -> list:


        """
        This function persforms a Range Query in the Rtree
        given a specified area

        Parameters:
        area: Is a list containing point_dim lists that each 
              contain 2 elements that are the lower and upper value of that dimension

        Returns:
        list: It returns a list with the points that the given rectangle contains
        """

        # this will contain all points found in range
        result = []

        # this is the list that contains all nodes left to check, it is a list whose 
        # elements are block ids of nodes that remain to be checked
        nodes_to_check = [self.root.block_id]
        current_node: Block_Indexfile | None = None

        while nodes_to_check: # ...is not empty

            current_node_to_check = nodes_to_check.pop()
            # Load the current block from disk
            # when current node is the root, no need to load it
            if current_node_to_check[0] != self.root.block_id: # ? why it has brackets given the fact that its an integer?

                # load node
                current_node = block_load_indexfile(
                    self.index_file_name, 
                    self.block_id_to_file_offset[current_node_to_check[0]]
                )
            else:
                current_node = self.root


            # check whether current node is leaf node
            if current_node.is_leaf:
                
                # check if any of the points contained in current node
                # are inside the query area
                for i_bl in range(current_node.size):
                    if self._rectangle_contains_point(
                        current_node.records[i_bl].vec,
                        area
                    ):
                        result.append(
                            current_node.records[i_bl].datafile_record_stored
                        )

            else:
                
                # check which entries (that are bounding boxes for below nodes)
                # intersect with query range
                for i_bl in range(current_node.size):
                    if self._rectangles_intersect(
                        current_node.records[i_bl].vec,
                        area
                    ):
                        nodes_to_check.append(
                            current_node.records[i_bl].datafile_record_stored
                        )

            # just to make clear that block is no longer needed in 
            # main memory
            del current_node

        return result
    
    def _is_dominated(self, e: Record_Indexfile, S: list[Record_Indexfile]):
        if not S:
            return False # because list is empty
        else:
                
            for point in S:
                if e.is_dominated(point):
                    return True
                
            return False
        
    def load_indexfile_block(self, b_id: int) -> Block_Indexfile:
        offset = self.block_id_to_file_offset(b_id)
        block = block_load_indexfile(self.index_file_name, offset)

        return block
                                
    def skyline_query(self) -> list[Record_Datafile]:

        """
        Executes a skyline query on the index file.

        This method retrieves the skyline points from the index file using a
        min-heap, using mindist as a key, to efficiently explore the records.
        It checks for domination among the records to determine the skyline.

        :return: A list of skyline points represented as `Record_Datafile`.
        :rtype: list[Record_Datafile]
        
        :raises SomeException: If there is an error while loading the index file
                            or during data file record retrieval.

        :example:

        Example usage of the skyline_query method:

        >>> skyline_points = instance.skyline_query()
        >>> print(skyline_points)  # Outputs the skyline points
        """

        S :list[Record_Indexfile] = list() # List of skyline points
        min_heap :list[Record_Indexfile] = list() # Heap of the items to be examined

        # Heap initialized with all the records of the root
        for child in self.root.records:
            heapq.heappush(min_heap, child) # this works fine because "less than" operator: __lt__() is implemented

        while min_heap:
            # e is an entry that it will either be a bounding box or a data point
            e: Record_Indexfile = heapq.heappop(min_heap)


            if self._is_dominated(e, S): # Check if e is dominated by any point in S
                continue
            else: # e is not dominated by any point in S
                if e.is_leaf:
                    S.append(e)
                else:
                    block = self.load_indexfile_block(e.datafile_record_stored)

                    for child in block.records:
                        if self._is_dominated(child, S) == False:
                            heapq.heappush(min_heap, child)

        points: list[Record_Datafile] = []
        for record in S:
            point = record_load_datafile(record.datafile_record_stored[0], record.datafile_record_stored[1])
            points.append(point)
        
        return points
    
    def _calculate_mindist(self, record: Record_Datafile | Record_Indexfile, point: list = None) -> int:

        if type(record) == Record_Indexfile:
            if record.is_leaf:
                converted: Record_Datafile = record_load_datafile(record.datafile_record_stored[0], record.datafile_record_stored[1])
                mindist = math.sqrt(sum((a-b)**2 for a, b in zip(converted.vec, point) ))
            else: # find mindist between an MBR and a point
                squared_distances = 0

                for dim in range(point_dim):
                    if point[dim] < record.vec[dim][0]:
                        squared_distances += (point[dim] - record.vec[dim][0])**2
                    elif point[dim] > record.vec[dim][1]:
                        squared_distances += (point[dim] - record.vec[dim][1])**2

                mindist =  math.sqrt(squared_distances)
        else:
            mindist = math.sqrt(sum((a-b)**2 for a, b in zip(record.vec, point) ))

        return mindist
        
    def knn_query(self, k: int, point: list) -> list[Record_Datafile]:


        """
        Executes a k-nearest neighbors (k-NN) query on the index structure.

        This method retrieves the `k` nearest records to a given point from the index file.
        It uses two heaps:
        - A min-heap (`min_heap`) to explore the child nodes based on the minimum distance.
        - A max-heap (`max_heap`) to maintain the `k` nearest records, using the negative of the distance as the key.

        The algorithm handles both leaf and inner nodes of the tree. If the root is a leaf, the method directly computes distances to the records. For inner nodes, it pushes the child nodes into the min-heap for further exploration.

        The max-heap is used to efficiently track the nearest neighbors, ensuring that the farthest neighbor is removed once `k` neighbors are found. The method also terminates early if it detects that no closer points can be found.

        :param k: The number of nearest neighbors to find.
        :type k: int
        :param point: The coordinates of the query point.
        :type point: list

        :return: A list of the `k` nearest records represented as `Record_Datafile`.
        :rtype: list[Record_Datafile]

        :raises SomeException: If there is an error while loading the index file
                            or during data file record retrieval.

        :example:

        Example usage of the knn method:

        >>> nearest_neighbors = instance.knn(5, [2.5, 4.3, 1.0])
        >>> print(nearest_neighbors)  # Outputs the 5 nearest neighbors
        """



        min_heap: list[Record_Indexfile] = list()
        max_heap: list[Record_Datafile] = list()

        # Min_heap initialized with all the records of the root
        # unless the root is a leaf which means use only max heap
        if self.root.is_leaf:
            for child in self.root.records:
                actual_record: Record_Datafile = record_load_datafile(child.datafile_record_stored[0], child.datafile_record_stored[1])
                mindist_of_actual_record: int = self._calculate_mindist(actual_record, point)

                if len(max_heap) == k:
                    heapq.heappushpop(max_heap, (-mindist_of_actual_record, actual_record))
                else:
                    heapq.heappush(max_heap, (-mindist_of_actual_record, actual_record)) # The negative is saved in max_heap because python only supports min heap
        else:

            for child in self.root.records:
                heapq.heappush(min_heap, (self._calculate_mindist(child, point), child)) # The child has the __lt__() implemented but it won't interfere as mindist has priority


        while min_heap:
            
            inner_record: Record_Indexfile = heapq.heappop(min_heap)

            # This check if there are possible closest point to the target
            # and if not then there is no reason to continue as we have
            # found the k nearest neighbors
            if len(max_heap) == k:
                mindist_of_top: Record_Datafile = -max_heap[0][0]

                if mindist_of_top < self._calculate_mindist(inner_record, point): 
                    min_heap.clear()
                    continue

            offset = self.block_id_to_file_offset(inner_record.datafile_record_stored)
            block: Block_Indexfile = block_load_indexfile(self.index_file_name, offset)

        
            if block.is_leaf :
                for child in block.records:
                    actual_record: Record_Datafile = record_load_datafile(child.datafile_record_stored[0], child.datafile_record_stored[1])
                    mindist_of_actual_record: int = self._calculate_mindist(actual_record, point)

                    if len(max_heap) == k:
                        heapq.heappushpop(max_heap, (-mindist_of_actual_record, actual_record))
                    else:
                        heapq.heappush(max_heap, (-mindist_of_actual_record, actual_record))

            else: # is inner block
                for child in block.records:
                    heapq.heappush(min_heap, (self._calculate_mindist(child, point), child))
        
        return max_heap

    def _shift_decimal_to_integer(self, num):
        # It shifts the decimal point of the num and 
        # eventually turning it an integer. The shift is 
        # always 7 positions because thats how many 
        # decimals geolocation coordinates need
        shifted_num = int(num * 10**7)
        
        return shifted_num

    def interleave_bits(self, point: list[int]):
        # Determine the maximum number of bits needed
        max_bits = max(point).bit_length()
        
        # Create a variable to hold the interleaved bits
        z_value = 0

        # Iterate over each bit position
        for bit_pos in range(max_bits):
            for i, num in enumerate(point):
                # Extract the bit at bit_pos from the current number
                bit = (num >> bit_pos) & 1
                # Place it in the correct position in the z-value
                z_value |= (bit << (bit_pos * len(point) + i))

        return z_value
    
    def get_z_value(self, rec: Record_Indexfile) -> int:
        integerized_coords = []

        for coordinate in rec.vec:
            integerized_coords.append(self._shift_decimal_to_integer(coordinate)) 


        z_value = self.interleave_bits(integerized_coords)

        return z_value
    
    # It performs two way merging between the blocks
    # in tuple1 and tuple2 
    def two_way_merge(self, tuple1, tuple2):
        
        free_ids = []
        free_ids_counter = 0

        counter1 = 0
        counter2 = 0

        block = block_load_indexfile(self.index_file_name,
                                       self.block_id_to_file_offset[tuple1[counter1]],
                                       tuple1[counter1])
        bucket1 = block.records[:block.size]

        # if block.block_id == 10:
        #     counter = 0
        #     for x in block.records[:block.size]:
        #         print(x.vec)
        #         counter += 1
        #     print('block 10: ', block.size, len(block.records), counter)

        free_ids.append(tuple1[counter1])
        

        counter1 += 1
        bucket1_counter = 0


        block = block_load_indexfile(self.index_file_name,
                                       self.block_id_to_file_offset[tuple2[counter2]],
                                       tuple2[counter2])
        bucket2 = block.records[:block.size]



       
            

        free_ids.append(tuple2[counter2])
        

        counter2 += 1
        bucket2_counter = 0

        filling_percentage = 0.7*block.max_num_of_records

        out = []

        remaining_bucket = 0

        # While elements in tuple1 and tuple2 
        while bucket1 and bucket2:

            # Take an element from each bucket
            rec1 = bucket1[bucket1_counter]
            rec2 = bucket2[bucket2_counter]

            


            # Compare the 2 elements by their z_values
            # and put the smallest in out block
            if self.get_z_value(rec1) < self.get_z_value(rec2):
                out.append(rec1)
                bucket1_counter += 1
            else:
                out.append(rec2)
                bucket2_counter += 1

            # If out block is filled up to the desirable
            # level, convert into a block and store it in
            # a free spot. Then empty out block to ready it
            # for the next round
            if len(out) >= filling_percentage:
                
                block = Block_Indexfile(point_dim,
                                        True,
                                        struct.calcsize(record_fmt_indexfile_leaf),
                                        free_ids[free_ids_counter])
               
                free_ids_counter += 1

                for rec in out:
                    block.add_record(rec)

                block_write_indexfile(block,
                                      self.index_file_name,
                                      self.block_id_to_file_offset[block.block_id])

                out = []

            
            if bucket1_counter >= len(bucket1): # if no more records in bucket1  
                if counter1 >= len(tuple1): # and no more blocks in tuple1
                    remaining_bucket = 2 # bucket2 has still elements to add
                    bucket1 = []
                else: # load a block in bucket1
                    block = block_load_indexfile(self.index_file_name,
                                       self.block_id_to_file_offset[tuple1[counter1]],
                                       tuple1[counter1])

                    bucket1 = block.records[:block.size]
                    free_ids.append(tuple1[counter1])
                    

                    counter1 += 1
                    bucket1_counter = 0

            if bucket2_counter >= len(bucket2): # if no more records in bucket2  
                if counter2 >= len(tuple2): # and no more blocks in tuple2
                    remaining_bucket = 1 # bucket1 has still elements to add
                    bucket2 = []
                else: # load a block in bucket2
                    block = block_load_indexfile(self.index_file_name,
                                       self.block_id_to_file_offset[tuple2[counter2]],
                                       tuple2[counter2])

                    bucket2 = block.records[:block.size]
                    free_ids.append(tuple2[counter2])
                 


                    counter2 += 1
                    bucket2_counter = 0


            
        
        # When tuple1 or tuple2 have no elements left then
        # the rest of the elements from the other tuple will
        # be arranged in blocks
        if remaining_bucket == 1:
            while bucket1:
                rec1 = bucket1[bucket1_counter]
                out.append(rec1)
                bucket1_counter += 1

                if len(out) >= filling_percentage:
                    
                    block = Block_Indexfile(point_dim,
                                            True,
                                            struct.calcsize(record_fmt_indexfile_leaf),
                                            free_ids[free_ids_counter])
                    
                    free_ids_counter += 1
                    for rec in out:
                        block.add_record(rec)

                    block_write_indexfile(block,
                                        self.index_file_name,
                                        self.block_id_to_file_offset[block.block_id])
                    

                    out = []

                if bucket1_counter >= len(bucket1): # if no more records in bucket1  
                    if counter1 >= len(tuple1): # and no more blocks in tuple1
                        
                        bucket1 = []

                        if(len(out) != 0):

                            block = Block_Indexfile(point_dim,
                                                True,
                                                struct.calcsize(record_fmt_indexfile_leaf),
                                                free_ids[free_ids_counter])
                        
                            free_ids_counter += 1
                            for rec in out:
                                block.add_record(rec)

                            block_write_indexfile(block,
                                                self.index_file_name,
                                                self.block_id_to_file_offset[block.block_id])
                            

                            out = []

                    else: # load a block in bucket1
                        block = block_load_indexfile(self.index_file_name,
                                        self.block_id_to_file_offset[tuple1[counter1]],
                                        tuple1[counter1])
                        

                        bucket1 = block.records[:block.size]
                        free_ids.append(tuple1[counter1])

                        

                        counter1 += 1
                        bucket1_counter = 0

        elif remaining_bucket == 2:
            while bucket2:
                rec2 = bucket2[bucket2_counter]
                out.append(rec2)
                bucket2_counter += 1

                if len(out) >= filling_percentage:
                    block = Block_Indexfile(point_dim,
                                            True,
                                            struct.calcsize(record_fmt_indexfile_leaf),
                                            free_ids[free_ids_counter])
                    free_ids_counter += 1

                    for rec in out:
                        block.add_record(rec)

                    block_write_indexfile(block,
                                        self.index_file_name,
                                        self.block_id_to_file_offset[block.block_id])
        
                    
                    out = []

                if bucket2_counter >= len(bucket2): # if no more records in bucket2
                    if counter2 >= len(tuple2): # and no more blocks in tuple2
                        bucket2 = []

                        if(len(out) != 0):

                            block = Block_Indexfile(point_dim,
                                                True,
                                                struct.calcsize(record_fmt_indexfile_leaf),
                                                free_ids[free_ids_counter])
                        
                            free_ids_counter += 1
                            for rec in out:
                                block.add_record(rec)

                            block_write_indexfile(block,
                                                self.index_file_name,
                                                self.block_id_to_file_offset[block.block_id])
                            

                            out = []
                
                    else: # load a block in bucket2
                        block = block_load_indexfile(self.index_file_name,
                                        self.block_id_to_file_offset[tuple2[counter2]],
                                        tuple2[counter2])

                        bucket2 = block.records[:block.size]
                        free_ids.append(tuple2[counter2])
                        

                        counter2 += 1
                        bucket2_counter = 0
        
        
        j = 0
        additive = min(free_ids)

        while j < len(free_ids) - 1:
            if j+additive == free_ids[j]:
                j += 1
            else:
                self.block_id_to_file_offset[-1] = self.block_id_to_file_offset.pop(j+additive)
                self.block_id_to_file_offset[j+additive] = self.block_id_to_file_offset.pop(free_ids[j])
                self.block_id_to_file_offset[free_ids[j]] = self.block_id_to_file_offset.pop(-1)

                j += 1


        
        

            
            


        
    
    def _external_mergesort(self):
        
        # Internally sort every block in indexfile based on z_value
        for b_id, offset in self.block_id_to_file_offset.items():
            block: Block_Indexfile = block_load_indexfile(self.index_file_name,
                                                          offset,
                                                          b_id)
            

            z_order_list = []
            for i in range(block.size):
                z_order_list.append( (self.get_z_value(block.records[i]), block.records[i]) )

            sorted_list: list[(int,Record_Indexfile)]= sorted(z_order_list, key = lambda x:x[0])
            
            
            block.id_to_index = {}
            for i in range(len(sorted_list)):
                block.records[i] = sorted_list[i][1]
        
            
                block.id_to_index[sorted_list[i][1].record_id] = i
            

            block_write_indexfile(block, 
                                  self.index_file_name,
                                  offset)
            

        # Externally sort every block in indexfile
        list_of_b_ids = [(x,) for x in self.block_id_to_file_offset.keys()]

        while len(list_of_b_ids) > 1:
            new_list = []
            for i in range(0, len(list_of_b_ids), 2):
                tuple1 = list_of_b_ids[i]
                tuple2 = list_of_b_ids[i+1] if i+1 < len(list_of_b_ids) else None
                
                if tuple2:
                    self.two_way_merge(tuple1, tuple2)
                    new_list.append(tuple1 + tuple2)
                else:
                    new_list.append(tuple1)

            list_of_b_ids = new_list
           

        
    def bulk_loading(self, datafile_name, df_blocks_offsets):
        
        # Bring all data to indexfile blocks
        recs_to_insert = list()
        num_of_df_blocks = len(df_blocks_offsets)
        offset_iterator = 0

        
        df_block = block_load_datafile(datafile_name, df_blocks_offsets[offset_iterator])

        offset_iterator +=1

        for rec in df_block.records:
            if rec.vec != [0,0]:
                recs_to_insert.append( (df_block.block_id, rec) )

            
            
        

        while recs_to_insert or offset_iterator < num_of_df_blocks:
            
            idxf_block: Block_Indexfile = Block_Indexfile(point_dim,
                                                          True,
                                                          struct.calcsize(record_fmt_indexfile_leaf),
                                                          self._give_next_available_block_id())
            while idxf_block.size < 0.7 * idxf_block.max_num_of_records:
                if len(recs_to_insert) == 0: # if list of recs is empty
                    if offset_iterator >= num_of_df_blocks: # and blocks in datafile are exhausted, terminate
                        break
                    else:
                        df_block = block_load_datafile(datafile_name, df_blocks_offsets[offset_iterator])
                        offset_iterator += 1

                        # test_counter = 0
                        for rec in df_block.records:
                            if rec.vec != [0,0]:
                                recs_to_insert.append( (df_block.block_id, rec) )

                        if len(recs_to_insert) == 0 :
                            break

                        #     if rec.vec == [0,0]:
                        #         test_counter+=1
                        #         print('papariaaaaaaaaaaaaaa ',df_block.block_id)
                        # print(test_counter)

                b_id, df_record = recs_to_insert.pop()


                idxf_record: Record_Indexfile = Record_Indexfile(dim= df_record.dim,
                                                                 is_leaf= True,
                                                                 datafile_record_stored= [b_id, df_record.record_id], 
                                                                 record_id= idxf_block.give_next_available_record_id(),
                                                                 vec= df_record.vec)
                idxf_block.add_record(idxf_record)
            

            self.block_id_to_file_offset[idxf_block.block_id] = self.offset_for_next_block_to_enter
            self. offset_for_next_block_to_enter = \
                block_write_indexfile(idxf_block, 
                                      self.index_file_name, 
                                      self.offset_for_next_block_to_enter
                                      )
            
        ###############################################
        # for b_id, offset in self.block_id_to_file_offset.items():
        #     block = block_load_indexfile(self.index_file_name,
        #                                  offset,
        #                                  b_id)
        #     flag = True
        #     counter = 0
        #     for x in block.records[:block.size]:
        #         if x.vec == [0,0]:
        #             counter += 1
        #             if flag:
        #                 print('ERRORRRRRRRRRRRR: ',block)
        #                 flag = False
        #     print('counter of false: ',counter)
            
        
        self._external_mergesort()

        ################################################
        # print("blocks after transfer")
        # for id,offset in self.block_id_to_file_offset.items():
        #     block = block_load_indexfile(self.index_file_name,
        #                                  offset,
        #                                  id)
        #     print(block)
        #     if block.block_id == 1:
        #         for i in range(block.size):
        #             print(block.records[i].vec)

        # sys.exit()

        

        # for i in range (1,11):
        #     block = block_load_indexfile(self.index_file_name,
        #                                 self.block_id_to_file_offset[i],
        #                                 i)
        
        #     mo_lat_1 = 0
        #     mo_lon_1 = 0
        #     for i in range(block.size):
        #         # print(block.records[i].vec[0],block.records[i].vec[1])
        #         mo_lat_1 += block.records[i].vec[0]/block.size
        #         mo_lon_1 += block.records[i].vec[1]/block.size

        #     print(mo_lat_1, mo_lon_1)

       



        
            
        
        pass

    def remove_point(self, record: Record_Datafile):
        pass


def rtree_write_indexfile(rtree: Rtree, indexfile_name):
    indexfile = open(indexfile_name, 'r+b')
    indexfile.seek(0, 2)
    rtree_indexfile_offset = indexfile.tell()
    indexfile.seek(0, 0)
    packed = struct.pack('>I', rtree_indexfile_offset)
    indexfile.write(packed)
    indexfile.seek(0, 2)

    # Packing and Writing the basic fields of R-Tree
    packed = struct.pack(
        rtree_fmt, 
        rtree.num_of_blocks,
        rtree.num_of_leaves,
        rtree.height,
        rtree.maximum_num_of_records,
        rtree.minimum_num_of_records,
        rtree.block_id_index_counter,
        rtree.block_id_to_file_offset[
            rtree.root.block_id
        ],
        rtree.forced_reinsert_enable
    )
    indexfile.write(packed)

    # Packing and Writing the root MBR
    args = [rtree.root_bounding_box[i][j] \
                for i in range(point_dim) \
                    for j in range(2)]
    packed = struct.pack(
        '>' + ''.join(['d' for _ in range(2 * point_dim)]),
        *args
    )
    indexfile.write(packed)

    # Packing and Writing the dictionary
    for k, v in rtree.block_id_to_file_offset.items():
        packed = struct.pack(
            '>IQ',
            k, v
        )
        indexfile.write(packed)
    indexfile.close()


def rtree_read_indexfile(indexfile_name) -> Rtree:
    with open(indexfile_name, 'r+b') as indexfile:

        # Moving the pointer into position, reading data, unpacking data

        # ...for the position of R-Tree in the indexfile
        indexfile.seek(0, 0)
        data_read = indexfile.read(struct.calcsize('>I'))
        rtree_indexfile_offset = struct.unpack('>I',data_read)

        # ...for the basic args
        indexfile.seek(rtree_indexfile_offset[0], 0)
        data_read = indexfile.read(struct.calcsize(rtree_fmt))
        first_args = struct.unpack(rtree_fmt, data_read)
            
        # ..for the bounding box
        format_for_root_bb = \
        '>' + ''.join(
            ['d' for _ in range(2 * point_dim)]
            )
        data_read = indexfile.read(struct.calcsize(format_for_root_bb))
        root_bb = struct.unpack(format_for_root_bb, data_read)
        root_bb = [
                [root_bb[i * 2 + j] for j in range(2)]
                    for i in range(point_dim)
        ]

        # ...for the dictionary
        block_id_to_file_offset = [] # Why is this a list and not a dict
        for _ in range(first_args[0]):
            block_id_to_file_offset.append(
                struct.unpack(
                    '>IQ',
                    indexfile.read(
                        struct.calcsize(
                            '>IQ'
                        )
                    )
                )
            )



    return Rtree(
        indexfile_name,
        first_args[6],
        first_args[7],
        first_args[3],
        first_args[4],
        first_args[0],
        first_args[1],
        first_args[2],
        root_bb,
        dict(block_id_to_file_offset),
        first_args[5],
        rtree_indexfile_offset[0]
    )
