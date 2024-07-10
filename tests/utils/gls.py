import numpy as np                                                                               # Importing NumPy for numerical operations.
import json
import re
from .base import BaseInterpolation                                                              # Importing BaseInterpolation class from the base module.
from scipy.sparse import csr_matrix                                                              # Importing csr_matrix for efficient sparse matrix operations.

def save_json_single_line_lists(data, file_path, indent=4):
    # First, convert the data to a JSON string with the desired indentation
    json_str = json.dumps(data, indent=indent)
    
    # Use regex to replace newlines in lists with a single space
    json_str = re.sub(r'\[\s+([^\]]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', json_str)

    # Write the modified JSON string to the file
    with open(file_path, 'w') as f:
        f.write(json_str)

class GlsInterpolation(BaseInterpolation):                                                       # Defining GlsInterpolation class that inherits from BaseInterpolation.
    def __init__(self, mesh):                                                                    # Initialization method for the GlsInterpolation class.
        super().__init__(mesh)                                                                   # Calling the initializer of the parent BaseInterpolation class with the mesh.
        self.A = self._compute_diffusion_magnitude()                                             # Computing diffusion magnitude using a helper method.
        self.in_vols_pairs = None                                                                # Initializing internal volume pairs attribute.
        self.Ns = None                                                                           # Initializing normal vectors attribute.
        self.in_faces_map = None                                                                 # Initializing internal faces map attribute.

    def interpolate(self):                                                                       # Main method to perform interpolation.
        self._set_internal_vols_pairs()                                                          # Setting internal volume pairs for volumes sharing an internal face.
        self._set_normal_vectors()                                                               # Setting normal vectors for faces in the mesh.
        self._set_in_faces_map()                                                                 # Setting internal face mappings to indices.

        all_nodes = self.mesh.nodes.all[:]                                                       # Getting all nodes from the mesh.
        dirichlet_nodes_flag = self.mesh.dirichlet_nodes_flag[:].flatten()                       # Getting Dirichlet node flags.
        dirichlet_nodes = all_nodes[dirichlet_nodes_flag == 1]                                   # Identifying Dirichlet nodes.

        neumann_nodes_flag = self.mesh.neumann_nodes_flag[:].flatten()                           # Getting Neumann node flags.
        neumann_nodes = all_nodes[neumann_nodes_flag == 1]                                       # Identifying Neumann nodes.

        in_nodes = np.setdiff1d(
            np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=True),
            neumann_nodes, assume_unique=True)                                                   # Identifying internal nodes by excluding Dirichlet and Neumann nodes.

        if len(in_nodes) > 0:                                                                    # Checking if there are any internal nodes.
            ws, vols_idx = self._interpolate_internal_nodes(in_nodes)                            # Interpolating internal nodes.
            vols_idx_flat = np.concatenate(vols_idx)                                             # Flattening volume indices.
            ws_flat = np.concatenate(ws)                                                         # Flattening weights.

            ns = np.array([len(vols) for vols in vols_idx])                                      # Counting the number of volumes for each node.
            nodes_idx = np.repeat(in_nodes, ns)                                                  # Repeating internal nodes indices according to their counts.

            W_in = csr_matrix((ws_flat, (nodes_idx, vols_idx_flat)), shape=(
                len(all_nodes), len(self.mesh.volumes)))                                         # Creating a sparse matrix for internal nodes interpolation.
        else:
            W_in = csr_matrix((len(all_nodes), len(self.mesh.volumes)))                          # Creating an empty sparse matrix if no internal nodes.

        if len(neumann_nodes) > 0:                                                               # Checking if there are any Neumann nodes.
            ws, vols_idx, neumann_nodes_ws = self._interpolate_neumann_nodes(neumann_nodes)      # Interpolating Neumann nodes.
            vols_idx_flat = np.concatenate(vols_idx)                                             # Flattening volume indices.
            ws_flat = np.concatenate(ws)                                                         # Flattening weights.

            neu_ws = np.zeros(len(all_nodes))                                                    # Initializing Neumann weights array with zeros.
            neu_ws[neumann_nodes] = neumann_nodes_ws[:]                                          # Assigning Neumann weights to corresponding nodes.

            ns = np.array([len(vols) for vols in vols_idx])                                      # Counting the number of volumes for each Neumann node.
            nodes_idx = np.repeat(neumann_nodes, ns)                                             # Repeating Neumann nodes indices according to their counts.

            W_neu = csr_matrix((ws_flat, (nodes_idx, vols_idx_flat)), shape=(
                len(all_nodes), len(self.mesh.volumes)))                                         # Creating a sparse matrix for Neumann nodes interpolation.
        else:
            W_neu = csr_matrix((len(all_nodes), len(self.mesh.volumes)))                         # Creating an empty sparse matrix if no Neumann nodes.
            neu_ws = np.zeros(len(all_nodes))                                                    # Initializing Neumann weights array with zeros.

        W = W_in + W_neu                                                                         # Combining internal and Neumann nodes sparse matrices.

        return W, neu_ws                                                                         # Returning the combined interpolation matrix and Neumann weights.
    def _interpolate_internal_nodes(self, in_nodes):                                             # Method to interpolate internal nodes.
        vols_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 3)                                                                      # Getting volumes around internal nodes.
        faces_around_in_nodes = self.mesh.nodes.bridge_adjacencies(
            in_nodes, 1, 2)                                                                      # Getting faces around internal nodes.
        ws = []                                                                                  # Initializing list to store weights.
        log_data = {
            "targets": {}
        }
        idx = 0
        for v, KSetv, Sv in zip(in_nodes, vols_around_in_nodes, faces_around_in_nodes):          # Iterating over internal nodes.
            nK = len(KSetv)                                                                      # Number of volumes around node v.
            nS = len(Sv)                                                                         # Number of faces around node v.

            Mv = np.zeros((nK + 3 * nS, 3 * nK + 1))                                             # Initializing least squares matrix Mv.
            Nv = np.zeros((nK + 3 * nS, nK))                                                     # Initializing least squares matrix Nv.

            self._assemble_ls_matrices(Mv, Nv, v, KSetv, Sv)                                     # Assembling least squares matrices for node v.

            M = np.linalg.inv(Mv.T @ Mv) @ (Mv.T @ Nv)                                           # Solving least squares problem to obtain matrix M.
            w = M[-1, :]                                                                         # Extracting weights from the last row of M.
            ws.append(w)                                                                         # Appending weights to the list.
            target_log = log_data["targets"].setdefault(f"Target {idx}", {
                "point": int(v),
                "KSetv": np.around(KSetv).tolist(),
                "Sv": np.around(Sv).tolist(),
                "Mv": np.around(Mv, decimals=2).tolist(),
                "Nv": np.around(Nv, decimals=2).tolist(),
                "weights": np.around(w, decimals=2).tolist()
            })
            idx += 1
        save_json_single_line_lists(log_data, 'log.json')
        return ws, vols_around_in_nodes                                                          # Returning weights and volume adjacencies.

    def _interpolate_neumann_nodes(self, neumann_nodes):                                         # Method to interpolate Neumann boundary nodes.
        vols_around_neu_nodes = self.mesh.nodes.bridge_adjacencies(
            neumann_nodes, 1, 3)                                                                 # Getting volumes around Neumann nodes.

        in_faces = self.mesh.faces.internal[:]                                                   # Getting internal faces.
        neumann_faces_flag = self.mesh.neumann_faces[:].flatten()                                # Getting Neumann face flags.
        neumann_faces = self.mesh.faces.all[neumann_faces_flag == 1]                             # Identifying Neumann faces.
        gN = self.mesh.neumann[:].flatten()                                                      # Getting Neumann boundary condition values.

        faces_around_neu_nodes = self.mesh.nodes.bridge_adjacencies(
            neumann_nodes, 1, 2)                                                                 # Getting faces around Neumann nodes.
        in_faces_around_neu_nodes = [
            np.intersect1d(f, in_faces, assume_unique=True)
            for f in faces_around_neu_nodes]                                                     # Getting internal faces around Neumann nodes.
        neu_faces_around_neu_nodes = [
            np.intersect1d(f, neumann_faces, assume_unique=True)
            for f in faces_around_neu_nodes]                                                     # Getting Neumann faces around Neumann nodes.

        ws = []                                                                                  # Initializing list to store weights.
        neu_ws = []                                                                              # Initializing list to store Neumann weights.

        for v, KSetv, Sv, Svb in zip(
                neumann_nodes, vols_around_neu_nodes,
                in_faces_around_neu_nodes, neu_faces_around_neu_nodes):                          # Iterating over Neumann nodes.
            nK = len(KSetv)                                                                      # Number of volumes around node v.
            nS = len(Sv)                                                                         # Number of internal faces around node v.
            nb = len(Svb)                                                                        # Number of Neumann faces around node v.

            Mb = np.zeros((nK + 3 * nS + nb, 3 * nK + 1))                                        # Initializing least squares matrix Mb.
            Nb = np.zeros((nK + 3 * nS + nb, nK + 1))                                            # Initializing least squares matrix Nb.

            self._assemble_ls_matrices(Mb, Nb, v, KSetv, Sv)                                     # Assembling least squares matrices for node v.

            neu_rows = np.arange(start=nK + 3 * nS, stop=nK + 3 * nS + nb)                       # Defining Neumann rows.

            Ks_Svb = self.mesh.faces.bridge_adjacencies(Svb, 2, 3).flatten()                     # Getting volumes associated with Neumann faces.
            sorter = np.argsort(KSetv)                                                           # Sorting volume indices.
            Ik = sorter[np.searchsorted(KSetv, Ks_Svb, sorter=sorter)]                           # Getting sorted indices for volumes.

            N_svb = self.Ns[Svb]                                                                 # Getting normal vectors for Neumann faces.
            L = self.mesh.permeability[Ks_Svb].reshape((nb, 3, 3))                               # Reshaping the permeability tensor for Neumann faces.
            nL = np.einsum("ij,ikj->ik", N_svb, L)                                               # Computing the dot product of normal vectors and permeability tensor.

            Mb[neu_rows, 3 * Ik] = -nL[:, 0]                                                     # Assembling the least squares matrix Mb for the x-component.
            Mb[neu_rows, 3 * Ik + 1] = -nL[:, 1]                                                 # Assembling the least squares matrix Mb for the y-component.
            Mb[neu_rows, 3 * Ik + 2] = -nL[:, 2]                                                 # Assembling the least squares matrix Mb for the z-component.

            Nb[neu_rows, nK] = gN[Svb]                                                           # Setting up the Nb matrix with Neumann boundary condition values.

            M = np.linalg.inv(Mb.T @ Mb) @ (Mb.T @ Nb)                                           # Solving the least squares problem to obtain matrix M.
            w = M[-1, :]                                                                         # Extracting weights from the last row of M.

            wi = w[:-1]                                                                          # Extracting internal weights.
            wc = w[-1]                                                                           # Extracting Neumann weights.

            ws.append(wi)                                                                        # Appending internal weights to the list.
            neu_ws.append(wc)                                                                    # Appending Neumann weights to the list.

        return ws, vols_around_neu_nodes, neu_ws                                                 # Returning weights, volume adjacencies, and Neumann weights.

    def _assemble_ls_matrices(self, Mi, Ni, v, KSetv, Sv):                                       # Method to assemble least squares matrices.
        nK = len(KSetv)                                                                          # Number of volumes around node v.
        nS = len(Sv)                                                                             # Number of faces around node v.

        xv = self.mesh.nodes.coords[v]                                                           # Coordinates of the node v.
        xK = self.mesh.volumes.center[KSetv]                                                     # Centers of volumes around node v.
        dKv = xK - xv                                                                            # Difference vectors between volume centers and node v.

        KSetV_range = np.arange(nK)                                                              # Range of indices for volumes.
        Mi[KSetV_range, 3 * KSetV_range] = dKv[:, 0]                                             # Setting x-component differences in Mi.
        Mi[KSetV_range, 3 * KSetV_range + 1] = dKv[:, 1]                                         # Setting y-component differences in Mi.
        Mi[KSetV_range, 3 * KSetV_range + 2] = dKv[:, 2]                                         # Setting z-component differences in Mi.
        Mi[KSetV_range, 3 * nK] = 1.0                                                            # Setting the last column in Mi to 1.0.

        Ni[KSetV_range, KSetV_range] = 1.0                                                       # Setting diagonal elements in Ni to 1.0.

        Sv_in_idx = self.in_faces_map[Sv]                                                        # Mapping internal faces to indices.
        Sv_in_idx = Sv_in_idx[Sv_in_idx != -1]                                                   # Filtering valid internal face indices.
        Ks_Sv = self.in_vols_pairs[Sv_in_idx, :]                                                 # Getting volume pairs for internal faces.
        sorter = np.argsort(KSetv)                                                               # Sorting volume indices.
        Ij1 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 0], sorter=sorter)]                         # Getting sorted indices for first volumes.
        Ij2 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 1], sorter=sorter)]                         # Getting sorted indices for second volumes.

        xS = self.mesh.faces.center[Sv]                                                          # Centers of faces around node v.
        eta_j = np.max(self.A[Ks_Sv], axis=1)                                                    # Maximum anisotropy magnitudes for faces.
        N_sj = self.Ns[Sv]                                                                       # Normal vectors for faces around node v.
        T_sj1 = xv - xS                                                                          # Difference vectors between node v and face centers.
        T_sj2 = np.cross(N_sj, T_sj1)                                                            # Cross product of normal vectors and T_sj1.
        tau_j2 = np.linalg.norm(T_sj2, axis=1) ** (-eta_j)                                       # Computing tau_j2 for anisotropy.
        tau_tsj2 = tau_j2[:, np.newaxis] * T_sj2                                                 # Scaling T_sj2 by tau_j2.

        L1 = self.mesh.permeability[Ks_Sv[:, 0]].reshape((nS, 3, 3))                             # Reshaping permeability tensor for first volumes.
        L2 = self.mesh.permeability[Ks_Sv[:, 1]].reshape((nS, 3, 3))                             # Reshaping permeability tensor for second volumes.
        nL1 = np.einsum("ij,ikj->ik", N_sj, L1)                                                  # Calculating normal permeability tensor product for first volumes.
        nL2 = np.einsum("ij,ikj->ik", N_sj, L2)                                                  # Calculating normal permeability tensor product for second volumes.

        idx1 = np.arange(start=nK, stop=nK + 3 * nS - 2, step=3)                                 # Indexing for first set of rows in Mi.
        idx2 = np.arange(start=nK + 1, stop=nK + 3 * nS - 1, step=3)                             # Indexing for second set of rows in Mi.
        idx3 = np.arange(start=nK + 2, stop=nK + 3 * nS, step=3)                                 # Indexing for third set of rows in Mi.

        Mi[idx1, 3 * Ij1] = -nL1[:, 0]                                                           # Setting x-component differences for first volumes in Mi.
        Mi[idx1, 3 * Ij1 + 1] = -nL1[:, 1]                                                       # Setting y-component differences for first volumes in Mi.
        Mi[idx1, 3 * Ij1 + 2] = -nL1[:, 2]                                                       # Setting z-component differences for first volumes in Mi.

        Mi[idx1, 3 * Ij2] = nL2[:, 0]                                                            # Setting x-component differences for second volumes in Mi.
        Mi[idx1, 3 * Ij2 + 1] = nL2[:, 1]                                                        # Setting y-component differences for second volumes in Mi.
        Mi[idx1, 3 * Ij2 + 2] = nL2[:, 2]                                                        # Setting z-component differences for second volumes in Mi.

        Mi[idx2, 3 * Ij1] = -T_sj1[:, 0]                                                         # Setting x-component differences in Mi.
        Mi[idx2, 3 * Ij1 + 1] = -T_sj1[:, 1]                                                     # Setting y-component differences in Mi.
        Mi[idx2, 3 * Ij1 + 2] = -T_sj1[:, 2]                                                     # Setting z-component differences in Mi.

        Mi[idx2, 3 * Ij2] = T_sj1[:, 0]                                                          # Setting x-component differences in Mi.
        Mi[idx2, 3 * Ij2 + 1] = T_sj1[:, 1]                                                      # Setting y-component differences in Mi.
        Mi[idx2, 3 * Ij2 + 2] = T_sj1[:, 2]                                                      # Setting z-component differences in Mi.

        Mi[idx3, 3 * Ij1] = -tau_tsj2[:, 0]                                                      # Setting x-component differences in Mi.
        Mi[idx3, 3 * Ij1 + 1] = -tau_tsj2[:, 1]                                                  # Setting y-component differences in Mi.
        Mi[idx3, 3 * Ij1 + 2] = -tau_tsj2[:, 2]                                                  # Setting z-component differences in Mi.

        Mi[idx3, 3 * Ij2] = tau_tsj2[:, 0]                                                       # Setting x-component differences in Mi.
        Mi[idx3, 3 * Ij2 + 1] = tau_tsj2[:, 1]                                                   # Setting y-component differences in Mi.
        Mi[idx3, 3 * Ij2 + 2] = tau_tsj2[:, 2]                                                   # Setting z-component differences in Mi.

    def _set_internal_vols_pairs(self):                                                          # Method to set pairs of volumes sharing an internal face.
        internal_faces = self.mesh.faces.internal[:]                                             # Getting all internal faces.
        self.in_vols_pairs = self.mesh.faces.bridge_adjacencies(
            internal_faces, 2, 3)                                                                # Setting volume pairs for internal faces.

    def _set_in_faces_map(self):                                                                 # Method to set the internal face map.
        internal_faces = self.mesh.faces.internal[:]                                             # Getting all internal faces.
        self.in_faces_map = np.zeros(len(self.mesh.faces), dtype=int)                            # Initializing the face map with zeros.
        self.in_faces_map[:] = -1                                                                # Setting all entries to -1 to indicate non-internal faces.
        self.in_faces_map[internal_faces] = np.arange(internal_faces.shape[0])                   # Mapping internal faces to their indices.

    def _set_normal_vectors(self):                                                               # Method to compute and store the normal vectors and their norms.
        all_faces = self.mesh.faces.all[:]                                                       # Getting all faces.

        faces_nodes = self.mesh.faces.bridge_adjacencies(all_faces, 0, 0)                        # Getting the nodes forming each face.
        I_idx = faces_nodes[:, 0]                                                                # Indices of the first nodes in each face.
        J_idx = faces_nodes[:, 1]                                                                # Indices of the second nodes in each face.
        K_idx = faces_nodes[:, 2]                                                                # Indices of the third nodes in each face.

        I = self.mesh.nodes.coords[I_idx]                                                        # Coordinates of the first nodes.
        J = self.mesh.nodes.coords[J_idx]                                                        # Coordinates of the second nodes.
        K = self.mesh.nodes.coords[K_idx]                                                        # Coordinates of the third nodes.

        Ns = np.cross(I - J, K - J)                                                              # Computing the normal vectors by taking the cross product.
        Ns_norm = np.linalg.norm(Ns, axis=1)                                                     # Computing the norms of the normal vectors.
        self.Ns = Ns / Ns_norm[:, np.newaxis]                                                    # Normalizing the normal vectors.

    def _compute_diffusion_magnitude(self):                                                      # Method to compute the diffusion magnitude.
        nvols = len(self.mesh.volumes)                                                           # Number of volumes in the mesh.
        Ks = self.mesh.permeability[:].reshape((nvols, 3, 3))                                    # Reshaping the permeability tensor.

        detKs = np.linalg.det(Ks)                                                                # Computing the determinants of the permeability tensor.
        trKs = np.trace(Ks, axis1=1, axis2=2)                                                    # Computing the traces of the permeability tensor.

        A = (1 - (3 * (detKs ** (1 / 3)) / trKs)) ** 2                                           # Computing the anisotropy measure.

        return A                                                                                 # Returning the diffusion magnitude.
