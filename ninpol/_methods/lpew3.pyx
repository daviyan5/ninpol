import numpy as np
from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from ..utils.robin_hood cimport unordered_map
from libc.stdio cimport printf
from libc.math cimport sqrt

DTYPE_I = int
DTYPE_F = float

cdef class LPEW3Interpolation:

    def __cinit__(self, int logging=False):
        self.logging = logging
        self.logger  = Logger("LPEW3")
        self.log_dict = {}

    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int dim = grid.dim
            int n_target = target_points.shape[0]
            int permeability_index = variable_to_index["cells"]["permeability"]
            int neumann_flag_index       = variable_to_index["points"]["neumann_flag" + "_" + variable]
            int neumann_val_index  = variable_to_index["points"]["neumann" + "_" + variable]

            DTYPE_F_t[:, :, ::1] permeability  = np.reshape(cells_data[permeability_index], 
                                                            (grid.n_elems, dim, dim))
            const DTYPE_I_t[::1] neumann_point = np.asarray(points_data[neumann_flag_index]).astype(DTYPE_I)
            const DTYPE_F_t[::1] neumann_val   = points_data[neumann_val_index]

        self.lpew3(grid, target_points, permeability, neumann_point, neumann_val, weights, neumann_ws)
    
    cdef void lpew3(self, Grid grid, const DTYPE_I_t[::1] target_points, DTYPE_F_t[:, :, ::1] permeability, 
                    const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                    DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int i
            int n_points = len(target_points)
            int point

            double sum_w    = 0.0
            double neu_term = 0.0

        for i in range(n_points):
            point = target_points[i]
            for j, elem in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point+1]]):
                weights[i, j] = 0.0
                weights[i, j] = self.partial_lpew3(grid, point, elem, permeability)
                sum_w = sum_w + weights[i, j]

            for j, elem in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point+1]]):
                weights[i, j] = weights[i, j] / sum_w

            if neumann_point[point]:
                neu_term = self.neumann_treatment(grid, point, neumann_val[point])
                neumann_ws[i] = neu_term

    cdef double partial_lpew3(self, Grid grid, int point, int elem, DTYPE_F_t[:, :, ::1] permeability):
        cdef:
            const DTYPE_I_t[::1] elem_faces  = grid.infael[elem]
            const DTYPE_I_t[::1] point_faces = grid.fsup[grid.fsup_ptr[point]:grid.fsup_ptr[point+1]]
            double zepta = 0.0
            double delta = 0.0

            double psi_fe, phi_fe, psi_e, phi_e, csi

            unordered_map[DTYPE_I_t, bool] face_map

        for face in elem_faces:
            face_map[face] = True
        
        for face in point_faces:
            if face_map.find(face) != face_map.end():
                psi_fe, phi_fe = 0.0, 0.0
                psi_e, phi_e   = 0.0, 0.0
                for face_elem in grid.esuf[grid.esuf_ptr[elem]:grid.esuf_ptr[elem+1]]:
                    if face_elem != elem:
                        psi_fe = self.psi_sum_lpew3(grid, point, face, face_elem, permeability)
                        phi_fe = self.phi_lpew3(grid, point, face, face_elem, permeability)
                csi   = self.csi_lpew3(grid, face, elem, permeability[elem])
                psi_e = self.psi_sum_lpew3(grid, point, face, elem, permeability)
                phi_e = self.phi_lpew3(grid, point, face, elem, permeability)
                zepta += (psi_e + psi_fe) * csi
                delta += (phi_fe + phi_e) * csi
        return zepta - delta

    cdef double neumann_treatment(self, Grid grid, int point, DTYPE_F_t neumann_val):
        pass      

    cdef double phi_lpew3(self, Grid grid, int point, int face, int elem, DTYPE_F_t[:, :, ::1] permeability):
        pass
    
    cdef double psi_sum_lpew3(self, Grid grid, int point, int face, int elem, DTYPE_F_t[:, :, ::1] permeability):
        pass
    
    cdef double volume(self, Grid grid, int face, DTYPE_I_t[::1] fpoints, DTYPE_F_t[::1] centroid):
        # Calculates the volume of the polyhedron formed by the points in fpoints + centroid - Is either a tetrahedron or a pyramid
        cdef:
            double face_area = grid.faces_areas[face]
            double face_center = grid.faces_centers[face]

            double height = sqrt((face_center[0] - centroid[0])**2 + 
                                 (face_center[1] - centroid[1])**2 + 
                                 (face_center[2] - centroid[2])**2)
        
        return face_area * height
            
        
        
        
        

    cdef double flux_term(self, DTYPE_F_t[::1] v1, DTYPE_F_t[:, ::1] K, DTYPE_F_t[::1] v2):
        pass

    cdef double lambda_lpew3(self, Grid grid, int point, int aux_point, int face, DTYPE_F_t[:, :, ::1] permeability):
        pass
    
    cdef double neta_lpew3(self, Grid grid, int point, int face, int elem, DTYPE_F_t[:, ::1] K):
        pass
    
    cdef double csi_lpew3(self, Grid grid, int face, int elem, DTYPE_F_t[:, ::1] K):
        cdef:
            double flux   = self.flux_term(grid.normal_faces[face], K)
            double volume = self.volume(grid, face, grid.inpofa[face], grid.centroids[elem])
            double csi    = flux / volume
        
        return csi
    
    cdef double sigma_lpew3(self, Grid grid, int point, int elem, DTYPE_F_t[:, :, ::1] permeability):
        pass

    