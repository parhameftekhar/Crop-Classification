import numpy as np
import torch
import torch.nn as nn
import itertools

class BaseSparseLoss(nn.Module):
    def __init__(self, img_height, img_width, window_size):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.window_size = window_size
        self.hash_to_index_vec = np.vectorize(self.hash_to_index)
        self.create_sparse_structure_from_images()

    def MortonFromPosition(self, position):
        """Convert integer (x,y,z) positions to Morton codes

        Args:
          positions: Nx3 np array (will be cast to int32)

        Returns:
          Length-N int64 np array
        """

        position = np.asarray(position, dtype=np.int32)
        morton_code = np.zeros(len(position), dtype=np.int64)
        coeff = np.asarray([4, 2, 1], dtype=np.int64)
        for b in range(21):
            morton_code |= ((position & (1 << b)) << (2 * b)) @ coeff
        assert morton_code.dtype == np.int64
        return morton_code

    def PositionFromMorton(self, morton_code):
        """Convert int64 Morton code to int32 (x,y,z) positions

        Args:
          morton_code: int64 np array

        Returns:
          Nx3 int32 np array
        """

        morton_code = np.asarray(morton_code, dtype=np.int64)
        position = np.zeros([len(morton_code), 3], dtype=np.int32)
        shift = np.array([2, 1, 0], dtype=np.int64)
        for b in range(21):
            position |= ((morton_code[:, np.newaxis] >> shift[np.newaxis, :]) >> (2 * b)
                         ).astype(np.int32) & (1 << b)
        assert position.dtype == np.int32
        return position

    def hash_to_index(self, hash_val, hash_table):
        if hash_val in hash_table:
            return hash_table[hash_val]
        else:
            return -1

    def create_sparse_structure_from_images(self):
        # CREATE NODES
        xindex, yindex = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
        xy_location = np.stack([yindex, xindex], axis=2).reshape(-1, 2)
        hash_code = self.MortonFromPosition(
            np.concatenate([xy_location, np.zeros((xy_location.shape[0], 1))], axis=1)
        )
        order = np.argsort(hash_code)

        ## MUCH REMEMBER ORDER
        xy_location = xy_location[order]
        hash_code = hash_code[order]
        hash_code_map = {code:i for i, code in enumerate(hash_code)}
            
        # ADD EDGES
        m = np.arange(self.window_size)-self.window_size//2
        edge_delta = np.array(
            list(itertools.product(m, m)),
            dtype=np.int32)
        max_edge_type = edge_delta.shape[0]

        #
        possible_node_i_indx = np.arange(xy_location.shape[0], dtype=np.int32)[:, np.newaxis] + np.zeros([1, max_edge_type], dtype=np.int32)
        possible_node_i_indx = possible_node_i_indx.flatten()
        possible_edge_types  = np.repeat(np.arange(0, max_edge_type).reshape(1, max_edge_type), xy_location.shape[0], axis=0).flatten()

        #
        possible_node_j_location = xy_location[:, np.newaxis, :] + edge_delta[np.newaxis, :, :]
        possible_node_j_location = possible_node_j_location.reshape([-1, 2])
        possible_node_j_hash = self.MortonFromPosition(
            np.concatenate([possible_node_j_location, np.zeros((possible_node_j_location.shape[0], 1))], axis=1)
        )
        possible_node_j_indx = self.hash_to_index_vec(possible_node_j_hash, hash_code_map)

        #
        valid_edges = possible_node_j_indx >= 0
        node_i_indx = possible_node_i_indx[valid_edges]
        node_j_indx = possible_node_j_indx[valid_edges]
        edges_type  = possible_edge_types[valid_edges]
        edges = np.stack([
            node_i_indx, node_j_indx
        ], axis=1)

        ## Control meta information here - Register as buffers to support .to(device)
        self.register_buffer('order', torch.from_numpy(order))
        self.register_buffer('node_locations', torch.from_numpy(xy_location))
        self.register_buffer('edges', torch.from_numpy(edges))
        self.register_buffer('edges_type', torch.from_numpy(edges_type))
