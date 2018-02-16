import numpy as np
import os, code, sys, time
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import chainer.functions as F
import tf_utils as utils

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# load data
#=============================================================================
#X_data = utils.load_datum(16, 0.6, normalize_data=True)
X_data = np.load('X16_06.npy')
K = 14
x = X_data[:8]




def alist_to_indexlist(alist):
    batch_size, N, K = alist.shape
    id1 = np.reshape(np.arange(batch_size),[batch_size,1])
    id1 = np.tile(id1,N*K).flatten()
    out = np.stack([id1,alist.flatten()], axis=1).astype(np.int32)
    return out

def get_kneighbor_alist(X_in, K=14, shell_fraction=0.1):
    batch_size, N, D = X_in.shape
    csr_list = get_csr_periodic_bc(X_in, K, shell_fraction=shell_fraction)
    adj_list = np.zeros((batch_size, N, K)).astype(np.int32)
    for i in range(batch_size):
        adj_list[i] = csr_list[i].indices.reshape(N, K)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    return alist_to_indexlist(adj_list)

def get_adjacency_list(X_in):
        """ search for K nneighbors, and return offsetted indices in adjacency list

        Args:
            X_in (numpy ndarray): input data of shape (mb_size, N, 6)
        """
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
        for i in range(mb_size):
            # this returns indices of the nn
            graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([N, K]) + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx
        return adj_list

def _get_status(coordinate, L_box, dL):
    """
    Assign a status to each coordinate (of a particle position inside the box):
    1 if 0 < coord < dL, 2 if L- dL < coord < L, 0 otherwise
    PARAMS:
        coordinate(float)
    RETURNS:
        status(int). Either 0, 1, or 2
    """
    if coordinate < dL:
        return 1
    elif L_box - dL < coordinate < L_box:
        return 2
    else:
        return 0

def _get_clone(particle, k, s, L_box, dL):
    """
    Clone a particle otuside of the box.
    PARAMS:
        particle(np array). 6-dim particle position in phase space
        k(int). Index of dimension that needs to be projected outside of the box.
        s(int). Status, either 1 or 2. Determines where should be cloned.
    RETURNS:
        clone(np array). 6-dim cloned particle position in phase space.
    """
    clone = []
    for i in range(6):
        if i == k:
            if s == 1:
                clone.append(particle[i] + L_box)
            elif s == 2:
                clone.append(particle[i] - L_box)
        else:
            clone.append(particle[i])
    return np.array(clone)

def get_pbc_adjacency_list(X_in, K, shell_fraction=0.1):
    """
    ORIGINAL
    """
    K = K
    mb_size, N, D = X_in.shape
    L_box = 16 if N == 16**3 else 32
    dL = L_box * shell_fraction
    box_size = (L_box, dL)
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        ids_map = {}  # For this batch will map new_id to old_id of cloned particles
        new_X = [part for part in X_in[i]]  # Start off with original cube
        for j in range(N):
            status = [_get_status(X_in[i, j, k], *box_size) for k in range(3)]
            if sum(status) == 0:  # Not in the shell --skip
                continue
            else:
                for k in range(3):
                    if status[k] > 0:
                        clone = _get_clone(X_in[i, j, :], k, status[k], *box_size)
                        new_X.append(clone)
                        ids_map.update({len(new_X) - 1: j})
                        for kp in range(k + 1, 3):
                            if status[kp] > 0:
                                bi_clone = _get_clone(clone, kp, status[kp], *box_size)
                                new_X.append(bi_clone)
                                ids_map.update({len(new_X) - 1: j})
                                for kpp in range(kp + 1, 3):
                                    if status[kpp] > 0:
                                        tri_clone = _get_clone(bi_clone, kpp, status[kpp], *box_size)
                                        new_X.append(tri_clone)
                                        ids_map.update({len(new_X) - 1: j})
        new_X = np.array(new_X)
        graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
        # Remap outbox neighbors to original ids
        for j in range(N):
            for k in range(K):
                if graph_idx[j, k] > N - 1:  # If outside of the box
                    graph_idx[j, k] = ids_map.get(graph_idx[j, k])
        graph_idx = graph_idx + (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list

def get_pbc_adjacency_list2(X_in, K, shell_fraction=0.1):
    """
    ORIGINAL
    """
    K = K
    mb_size, N, D = X_in.shape
    L_box = 16 if N == 16**3 else 32
    dL = L_box * shell_fraction
    box_size = (L_box, dL)
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        ids_map = {}  # For this batch will map new_id to old_id of cloned particles
        new_X = [part for part in X_in[i]]  # Start off with original cube
        for j in range(N):
            status = [_get_status(X_in[i, j, k], *box_size) for k in range(3)]
            if sum(status) == 0:  # Not in the shell --skip
                continue
            else:
                for k in range(3):
                    if status[k] > 0:
                        clone = _get_clone(X_in[i, j, :], k, status[k], *box_size)
                        new_X.append(clone)
                        ids_map.update({len(new_X) - 1: j})
                        for kp in range(k + 1, 3):
                            if status[kp] > 0:
                                bi_clone = _get_clone(clone, kp, status[kp], *box_size)
                                new_X.append(bi_clone)
                                ids_map.update({len(new_X) - 1: j})
                                for kpp in range(kp + 1, 3):
                                    if status[kpp] > 0:
                                        tri_clone = _get_clone(bi_clone, kpp, status[kpp], *box_size)
                                        new_X.append(tri_clone)
                                        ids_map.update({len(new_X) - 1: j})
        new_X = np.array(new_X)
        graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
        # Remap outbox neighbors to original ids
        '''
        for j in range(N):
            for k in range(K):
                if graph_idx[j, k] > N - 1:  # If outside of the box
                    graph_idx[j, k] = ids_map.get(graph_idx[j, k])
        graph_idx = graph_idx #+ (N * i)  # offset idx for batches
        '''
        graph_idx = graph_idx + (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list

def _truncateOG(csr, N, K, ids_map):
    graph_idx = csr.indices.reshape([-1, K]) # CONFIRMED
    graph_idx = graph_idx[:N, :]
    for j in range(N):
        for k in range(K):
            if graph_idx[j, k] > N - 1:  # If outside of the box
                print('truncOG: {}'.format(graph_idx[j,k]))
                code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
                graph_idx[j, k] = ids_map.get(graph_idx[j, k])
    return graph_idx

def _truncateLIL(csr, N, K, ids_map):
    graph_lil = csr[:N,:].tolil()
    for j in range(N):
        cur_row = graph_lil.rows[j]
        truncated_row = [r if r < N else ids_map[r] for r in cur_row]
        if cur_row != truncated_row:
            print('{} | {}'.format(cur_row, truncated_row))
            code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

        graph_lil.rows[j] = truncated_row
        #('{}'.format(graph_lil.rows[j]))
    graph_idx = graph_lil.tocsr().indices.reshape(N, K)
    return graph_idx

'''
mb_size = 10
for i in range(0, 1000 - mb_size, mb_size):
    x_in = X_data[i:i+mb_size]
    print('{}: {}'.format(i, x_in.shape))
    alist_og = get_pbc_adjacency_list(np.copy(x_in), K, shell_fraction=0.1)
    alist_2 = get_pbc_adjacency_list2(np.copy(x_in), K, shell_fraction=0.1)
    assert np.all(alist_og == alist_2)
'''

class KNN_v2():
    def __init__(self, X_in, K, L_box, shell_fraction=0.1):
        self.K = K
        self.L_box = L_box  # Box size

        # Change shell_fraction for tuning the thickness of the shell to be replicated, must be in range 0-1
        # 0: no replication, 1: entire box is replicated to the 26 neighbouring boxes
        self.shell_fraction = shell_fraction
        self.dL = self.L_box * self.shell_fraction

        self.adjacency_list_periodic_bc = self.get_adjacency_list_periodic_bc_v2(X_in)

    def _get_status(self, coordinate):
        """
        Assign a status to each coordinate (of a particle position inside the box):
        1 if 0 < coord < dL, 2 if L- dL < coord < L, 0 otherwise

        PARAMS:
            coordinate(float)
        RETURNS:
            status(int). Either 0, 1, or 2
        """
        if coordinate < self.dL:
            return 1
        elif self.L_box - self.dL < coordinate < self.L_box:
            return 2
        else:
            return 0

    def _get_clone(self, particle, k, s):
        """
        Clone a particle otuside of the box.

        PARAMS:
            particle(np array). 6-dim particle position in phase space
            k(int). Index of dimension that needs to be projected outside of the box.
            s(int). Status, either 1 or 2. Determines where should be cloned.
        RETURNS:
            clone(np array). 6-dim cloned particle position in phase space.
        """
        clone = []
        for i in range(6):
            if i == k:
                if s == 1:
                    clone.append(particle[i] + self.L_box)
                elif s == 2:
                    clone.append(particle[i] - self.L_box)
            else:
                clone.append(particle[i])
        return np.array(clone)


    def get_adjacency_list_periodic_bc_v2(self, X_in):
        """
        Map inner chunks to outer chunks
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)

        for i in range(mb_size):
            ids_map = {}  # For this batch will map new_id to old_id of cloned particles
            new_X = [part for part in X_in[i]]  # Start off with original cube
            for j in range(N):
                status = [self._get_status(X_in[i, j, k]) for k in range(3)]
                if sum(status) == 0:  # Not in the shell --skip
                    continue
                else:
                    for k in range(3):
                        if status[k] > 0:
                            clone = self._get_clone(particle=X_in[i, j, :], k=k, s=status[k])
                            new_X.append(clone)
                            ids_map.update({len(new_X) - 1: j})
                            for kp in range(k + 1, 3):
                                if status[kp] > 0:
                                    bi_clone = self._get_clone(particle=clone, k=kp, s=status[kp])
                                    new_X.append(bi_clone)
                                    ids_map.update({len(new_X) - 1: j})
                                    for kpp in range(kp + 1, 3):
                                        if status[kpp] > 0:
                                            tri_clone = self._get_clone(particle=bi_clone, k=kpp, s=status[kpp])
                                            new_X.append(tri_clone)
                                            ids_map.update({len(new_X) - 1: j})

            new_X = np.array(new_X)
            graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
            # Remap outbox neighbors to original ids
            for j in range(N):
                for k in range(K):
                    if graph_idx[j, k] > N - 1:  # If outside of the box
                        graph_idx[j, k] = ids_map.get(graph_idx[j, k])

            graph_idx = graph_idx + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx

        return adj_list

class KNN():
    def __init__(self, X_in, K, L_box):
        self.K = K
        self.L_box = L_box  # Box size

        # Change shell_fraction for tuning the thickness of the shell to be replicated, must be in range 0-1
        # 0: no replication, 1: entire box is replicated to the 26 neighbouring boxes
        self.shell_fraction = 0.1
        self.dL = self.L_box * self.shell_fraction

        # self.adjacency_list = self.get_adjacency_list(X_in)
        self.adjacency_list = self.get_adjacency_list_periodic_bc(X_in)

    def get_adjacency_list(self, X_in):
        """ search for K nneighbors, and return offsetted indices in adjacency list

        Args:
            X_in (numpy ndarray): input data of shape (mb_size, N, 6)
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
        for i in range(mb_size):
            # this returns indices of the nn
            graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([N, K]) + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx
        return adj_list

    def get_adjacency_list_periodic_bc(self, X_in):
        """
        Same as get_adjacency_list but includes periodic boundary conditions.

        PARAMS:
            X_in(numpy ndarray). Input data of shape (mb_size, N, 6).

        RETURNS:
            adj_list(numpy ndarray). Adjacency list of shape (mb_size, N, K)
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
        for i in range(mb_size):
            ids_map = {}  # For this batch will map new_id to old_id of cloned particles
            new_X = [part for part in X_in[i]]  # Start off with original cube
            for j in range(N):
                if self._is_in_inbox_shell(X_in[i, j, :3]):  # Consider only inbox particles in a shell close to the boundary
                    # Map this particle to the 26 neighbouring boxes, but keep only clones that leave in the
                    # corresponding outside shell close to the boundary.
                    for x0 in [X_in[i, j, 0], X_in[i, j, 0] + self.L_box, X_in[i, j, 0] - self.L_box]:
                        for x1 in [X_in[i, j, 1], X_in[i, j, 1] + self.L_box, X_in[i, j, 1] - self.L_box]:
                            for x2 in [X_in[i, j, 2], X_in[i, j, 2] + self.L_box, X_in[i, j, 2] - self.L_box]:
                                # Check if it's in the outside shell and remove mapping to itself
                                if self._is_in_outbox_shell([x0, x1, x2]) and not self._is_in_box([x0, x1, x2]):
                                    new_X.append(np.array([x0, x1, x2, X_in[i, j, 3], X_in[i, j, 4], X_in[i, j, 5]]))
                                    ids_map.update({len(new_X) - 1: j}) # Update map {new_id: old_id}

            new_X = np.array(new_X)
            graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
            # Remap outbox neighbors to original ids
            for j in range(N):
                for k in range(K):
                    if graph_idx[j, k] > N - 1:  # If outside of the box
                        graph_idx[j, k] = ids_map.get(graph_idx[j, k])

            #print(graph_idx.shape)

            graph_idx = graph_idx + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx

        return adj_list

    # Bunch of helper methods
    def _is_in_inbox_shell(self, x):
        """
        x must be a particle INSIDE the original box, then returns true if it's in a cubic shell of thickness dL
        from the boundary of the box

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_shell(bool).
        """
        is_in_shell = False
        for i in range(3):
            if x[i] < self.dL or x[i] > self.L_box - self.dL:
                is_in_shell = True
                break
        return is_in_shell

    def _is_in_outbox_shell(self, x):
        """
        x must be a particle OUTSIDE the original box, then returns true if it's in a cubic shell of thickness dL
        from the boundary of the box

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_shell(bool).
        """
        is_in_shell = True
        for i in range(3):
            if x[i] > self.L_box + self.dL or x[i] < -self.dL:
                is_in_shell = False
                break
        return is_in_shell

    def _is_in_box(self, x):
        """
        x can be either outside or inside the original box. Returns true if it is inside the box.

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_box(bool).
        """
        is_in_box = True
        for i in range(3):
            if x[i] > self.L_box or x[i] < 0:
                is_in_box = False
                break
        return is_in_box

alist_og    = get_pbc_adjacency_list(np.copy(x), K, shell_fraction=0.1)#0.3)
alist_nopbc = get_adjacency_list(np.copy(x))
alist_pbc2   = KNN_v2(np.copy(x), K, 16).adjacency_list_periodic_bc
alist_pbc1   = KNN(np.copy(x), K, 16).adjacency_list
assert np.all(alist_og == alist_nopbc) and np.all(alist_og == alist_pbc2) and np.all(alist_pbc1 == alist_pbc2)# this should cause assertion error

'''
Well, the periodic boundary conditions not changing anything.

Likely because this line?:
'graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box'
'''
