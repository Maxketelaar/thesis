import numpy as np
import scipy as sp
import topogenesis as tg
import pyvista as pv
import trimesh as tm

def transform_mat(value):
    mat = np.identity(4)
    mat[:3,-1] = np.array(value)
    return mat

# ------------------------------------------------------ #
# convert trimesh object to pyvista object
# from: AnastasiaFlorou
# ------------------------------------------------------ #
def tri_to_pv(tri_mesh):
    faces = np.pad(tri_mesh.faces, ((0, 0),(1,0)), 'constant', constant_values=3)
    pv_mesh = pv.PolyData(tri_mesh.vertices, faces)
    return pv_mesh

# ------------------------------------------------------ #
# reshape and store values into envelope-shape lattice
# from: Topogenesis/SAzadadi
# ------------------------------------------------------ #
def reshape_and_store_to_lattice(values_list, envelope_lattice):
    env_all_vox_id = envelope_lattice.indices.flatten()
    env_all_vox = envelope_lattice.flatten() # envelope inclusion condition: True-False
    env_in_vox_id = env_all_vox_id[env_all_vox] # keep in-envelope voxels (True)

    # initialize array
    values_array = np.full(env_all_vox.shape, 0.0)
    
    # store values for the in-envelope voxels
    values_array[env_in_vox_id] = values_list

    # reshape to lattice shape
    values_array_3d = values_array.reshape(envelope_lattice.shape)

    # convert to lattice
    values_lattice = tg.to_lattice(values_array_3d, envelope_lattice)

    return values_lattice

# ------------------------------------------------------ #
# construct intervisibilities network/graph
# with regard to a visibility objective
# From: generative solar-climatic configurations by AFlorou
# ------------------------------------------------------ #
def construct_graph(reference_vectors, hitface_id, ray_id, envelope_lattice, faces_numb):

    # voxel centroids
    vox_cts = envelope_lattice.centroids_threshold(-0.1)

    # initialize array for inter-dependencies of voxels
    G = np.zeros((len(vox_cts),len(vox_cts),len(reference_vectors)))

    # initialize array of obstructed rays
    U = np.zeros((len(vox_cts),len(reference_vectors)))

    # how many faces each ray hits
    unq_rays, unq_counts = np.unique(ray_id, return_counts = True)

    # total number of faces of all the voxels
    vox_faces_tot = len(vox_cts)*faces_numb

    f0 = 0 # first face id

    # iterate through the rays
    for ray in unq_rays:
        # the faces that this ray hits
        faces = hitface_id[f0 : f0 + unq_counts[ray]]
        f0 += unq_counts[ray] # first face_id hit by the next ray

        # voxel from which the ray originates
        vox_id = np.floor(ray/len(reference_vectors)).astype(int)

        # vector/direction to which the ray corresponds
        vector = ray - vox_id*len(reference_vectors)
        # print("ray =", ray, "vox_id =", vox_id, "len(reference_vectors =", len(reference_vectors))
        # check if any of the hit face_id belongs to the context meshes
        c_faces = sum(f > vox_faces_tot for f in faces)

        if c_faces == 0: # if the ray did not hit the context mesh

            # find to which voxel each hit face belongs
            voxels = np.floor(faces/faces_numb).astype(int)

            # remove duplicates
            unq_voxs = np.unique(voxels)

            # remove source voxel
            blocking_voxs = np.delete(unq_voxs, np.where(unq_voxs == vox_id))

            # store the blocks for this voxel
            G[blocking_voxs, vox_id, vector] = 1
            # print("G", blocking_voxs, vox_id, vector)

        else:
            # store the obstructed ray
            U[vox_id, vector] = 1
            # print("U", vox_id, vector)
    
    return G, U

# ------------------------------------------------------ #
# store voxels' and faces' interdependencies (ordered) 
# with regard to a visibility objective
# From: generative solar-climatic configurations by AFlorou
# ------------------------------------------------------ #
def store_interdependencies(reference_vectors, hitface_id, ray_id, envelope_lattice, faces_number):

    # voxel centroids
    vox_cts = envelope_lattice.centroids

    # initialize array for inter-dependencies of voxels
    voxel_blocks = np.zeros((len(vox_cts),len(vox_cts),len(reference_vectors)))

    # initialize array for inter-dependencies of faces
    face_blocks = np.zeros((faces_number*len(vox_cts),faces_number*len(vox_cts),len(reference_vectors)))

    # how many faces each ray hits
    unq_rays, unq_counts = np.unique(ray_id, return_counts = True)

    # total number of faces of all the voxels
    vox_faces_tot = len(vox_cts)*faces_number

    f0 = 0 # first face id

    # iterate through the rays
    for ray in unq_rays:
        # the faces that this ray hits
        faces = hitface_id[f0 : f0 + unq_counts[ray]]
        f0 += unq_counts[ray] # first face_id hit by the next ray

        # check if any of the hit face_id belongs to the context meshes
        c_faces = sum(f > vox_faces_tot for f in faces)

        if c_faces == 0: # if the ray did not hit the context mesh

            # voxel from which the ray originates
            v_id = np.floor(ray/len(reference_vectors)).astype(int)
            
            # sun vector/direction to which the ray corresponds
            s_dir = ray - v_id*len(reference_vectors)

            # find to which voxel each hit face belongs
            voxels = np.floor(faces/faces_number).astype(int)

            # remove duplicates
            unq_voxs = np.unique(voxels)

            # extract centroids of these voxels
            v_cens = vox_cts[unq_voxs]

            # point source of ray
            source = vox_cts[v_id]

            # calculate distance of all the voxels' centroids from the ray source
            dists = [sp.spatial.distance.euclidean(source, c) for c in v_cens]
            dists_array = np.array(dists)

            # sort distances
            dists_sorted, vox_level = np.unique(dists_array, return_inverse = True)

            # add +1 so the original voxel is of level 1
            vox_level += 1

            # store the blocks for this voxel (ordered)
            voxel_blocks[v_id, unq_voxs, s_dir] = vox_level

            # find the index of the face of the source voxel
            ind = np.where(unq_voxs==v_id)[0]
            f1 = faces[ind]

            # store the blocks for this face (unordered)
            face_blocks[f1, faces, s_dir] = 1
        
    return voxel_blocks, face_blocks

# ------------------------------------------------------ #
# construct intervisibilities network/graph
# without taking own origins into account
# From: adapted
# ------------------------------------------------------ #
def ground_graph(reference_vectors, hitface_id, ray_id, envelope_lattice, faces_numb, ground_lattice):

    # voxel centroids
    vox_cts = envelope_lattice.centroids
    grnd_cts = ground_lattice.centroids
    
    # initialize array for inter-dependencies of voxels
    G = np.zeros((len(vox_cts),len(grnd_cts),len(reference_vectors)))

    # initialize array of obstructed rays
    U = np.zeros((len(grnd_cts),len(reference_vectors)))

    # how many faces each ray hits
    unq_rays, unq_counts = np.unique(ray_id, return_counts = True)

    # total number of faces of all the voxels
    vox_faces_tot = len(vox_cts)*faces_numb

    # first face id
    f0 = 0 

    # iterate through the rays
    for id, ray in enumerate(unq_rays):

        # the faces that this ray hits
        faces = hitface_id[f0 : f0 + unq_counts[id]]

        # first face_id hit by the next ray
        f0 += unq_counts[id] 

        # voxel from which the ray originates
        vox_id = np.floor(ray/len(reference_vectors)).astype(int)

        # vector/direction to which the ray corresponds
        vector = ray - vox_id*len(reference_vectors)

        # check if any of the hit face_id belongs to the context meshes
        c_faces = sum(f > vox_faces_tot for f in faces)

        # if the ray did not hit the context mesh:
        if c_faces == 0: 

            # find to which voxel each hit face belongs
            voxels = np.floor(faces/faces_numb).astype(int)

            # remove duplicates
            unq_voxs = np.unique(voxels)

            # store the blocks for this voxel
            G[unq_voxs, vox_id, vector] = 1 

        else:
            # store the obstructed ray
            U[vox_id, vector] = 1
    
    return G, U 

# ------------------------------------------------------ #
# find centroids of voxels with no neighbours in 
# certain directions
# From: adapted with help from Shervin for origin
# ------------------------------------------------------ #

def find_centroids(lattice, ref_lattice, dir, axes):

    # shifting all the voxels one level in a certain direction
    shifted_lattice_Y_pos = np.roll(lattice, (0,0,dir),axis=axes) 
    
    # an exposed facade surface exists where a voxel is filled (1) and the voxel next to it is empty (0)
    side_voxels_3darray_padded = (lattice == 1) *  (shifted_lattice_Y_pos == 0)

    # removing the pad
    side_voxels_3darray = side_voxels_3darray_padded[1:-1, 1:-1, 1:-1]
    
    # convert to lattice
    side_voxels_lattice = tg.to_lattice(side_voxels_3darray, ref_lattice)

    # extracting the centroids of all exposed voxels
    centroids = side_voxels_lattice.centroids

    return centroids

# ------------------------------------------------------ #
# constructing meshes for all 5 directions of a voxel 
# floor is not used
# From: adapted with help from Shervin for origin
# ------------------------------------------------------ #
 
def construct_mesh_y_pos(centroid, unit):
    meshes= []
    for i, cen in enumerate(centroid):
        # generating the vertices of the side faces in +Y direction
        # centroid + half of the unit size in the four top directions
        v0 = cen + 0.5 * unit * np.array([ 1, -1, 1]) # side right above
        v1 = cen + 0.5 * unit * np.array([ 1,-1, -1]) # side right below
        v2 = cen + 0.5 * unit * np.array([-1,-1, -1]) # side left below
        v3 = cen + 0.5 * unit * np.array([-1, -1, 1]) # side left above

        face_a = [v0,v1,v2] # trimesh only takes triangular meshes, no quad meshes
        face_b = [v2,v3,v0]

        mesh_a = tm.Trimesh(vertices= face_a, faces= [[0,2,1]])
        mesh_b = tm.Trimesh(vertices= face_b, faces= [[0,2,1]])

        meshes.append(mesh_a)
        meshes.append(mesh_b)

    return meshes

def construct_mesh_y_neg(centroid, unit):
    meshes= []
    for i, cen in enumerate(centroid):
        # generating the vertices of the side faces in +Y direction
        # centroid + half of the unit size in the four top directions
        v0 = cen + 0.5 * unit * np.array([ 1, 1, 1]) # side right above
        v1 = cen + 0.5 * unit * np.array([ 1,1, -1]) # side right below
        v2 = cen + 0.5 * unit * np.array([-1,1, -1]) # side left below
        v3 = cen + 0.5 * unit * np.array([-1, 1, 1]) # side left above

        face_a = [v1,v2,v3] # trimesh only takes triangular meshes, no quad meshes
        face_b = [v3,v0,v1]

        mesh_a = tm.Trimesh(vertices= face_a, faces= [[1,2,0]])
        mesh_b = tm.Trimesh(vertices= face_b, faces= [[1,2,0]])

        meshes.append(mesh_a)
        meshes.append(mesh_b)

    return meshes

def construct_mesh_x_pos(centroid, unit):
    meshes= []
    for i, cen in enumerate(centroid):
        # generating the vertices of the side faces in +Y direction
        # centroid + half of the unit size in the four top directions
        v0 = cen + 0.5 * unit * np.array([ -1, -1, 1]) # side right above
        v1 = cen + 0.5 * unit * np.array([ -1,-1, -1]) # side right below
        v2 = cen + 0.5 * unit * np.array([-1,1, -1]) # side left below
        v3 = cen + 0.5 * unit * np.array([-1, 1, 1]) # side left above

        face_a = [v0,v1,v2] # trimesh only takes triangular meshes, no quad meshes
        face_b = [v2,v3,v0]

        mesh_a = tm.Trimesh(vertices= face_a, faces= [[2,1,0]])
        mesh_b = tm.Trimesh(vertices= face_b, faces= [[2,1,0]])

        meshes.append(mesh_a)
        meshes.append(mesh_b)

    return meshes

def construct_mesh_x_neg(centroid, unit):
    meshes= []
    for i, cen in enumerate(centroid):
        # generating the vertices of the side faces in +Y direction
        # centroid + half of the unit size in the four top directions
        v0 = cen + 0.5 * unit * np.array([ 1, 1, 1]) # side right above
        v1 = cen + 0.5 * unit * np.array([1,1, -1]) # side right below
        v2 = cen + 0.5 * unit * np.array([1,-1, -1]) # side left below
        v3 = cen + 0.5 * unit * np.array([1, -1, 1]) # side left above

        face_a = [v0,v1,v2] # trimesh only takes triangular meshes, no quad meshes
        face_b = [v2,v3,v0]

        mesh_a = tm.Trimesh(vertices= face_a, faces= [[2,1,0]])
        mesh_b = tm.Trimesh(vertices= face_b, faces= [[2,1,0]])

        meshes.append(mesh_a)
        meshes.append(mesh_b)

    return meshes

def construct_mesh_z_pos(centroid, unit):
    meshes= []
    for i, cen in enumerate(centroid):
        # generating the vertices of the side faces in +Y direction
        # centroid + half of the unit size in the four top directions
        v0 = cen + 0.5 * unit * np.array([ 1, 1, 1]) # side right above
        v1 = cen + 0.5 * unit * np.array([ 1,-1, 1]) # side right below
        v2 = cen + 0.5 * unit * np.array([-1,-1, 1]) # side left below
        v3 = cen + 0.5 * unit * np.array([-1, 1, 1]) # side left above

        face_a = [v0,v1,v2] # trimesh only takes triangular meshes, no quad meshes
        face_b = [v2,v3,v0]

        mesh_a = tm.Trimesh(vertices= face_a, faces= [[0,1,2]])
        mesh_b = tm.Trimesh(vertices= face_b, faces= [[0,1,2]])

        meshes.append(mesh_a)
        meshes.append(mesh_b)

    return meshes

# ------------------------------------------------------ #
# constructing meshes and normals/points for each face 
# 
# From: adapted with help from Shervin for origin
# ------------------------------------------------------ #

# this function could be much shorter probably --> 4x almost identical operations, does works fine though

def construct_vertical_mesh(lat,unit):
    vertical_meshes = []
    test_points = []
    test_point_normals = []

    # padding to avoid the rolling issue
    input_lattice_padded = np.pad(lat, 1, mode='constant',constant_values=0)

    # Y_positive mesh
    Y_pos_centroids = find_centroids(lattice= input_lattice_padded, ref_lattice=lat, dir= 1, axes= 1)
    Y_pos_mesh = construct_mesh_y_pos(Y_pos_centroids, unit)
    vertical_meshes.extend(Y_pos_mesh)

    # move the test points so they are not inside mesh edge
    Y_pos_text_point = Y_pos_centroids + (lat.unit/2 + 0.01) * [0,-1,0]
    test_points.extend(Y_pos_text_point)

    # find normal of squares: take every other triangle normal
    Y_pos_mesh = tm.util.concatenate(Y_pos_mesh)
    Y_pos_normals = Y_pos_mesh.face_normals[::2]
    test_point_normals.extend(Y_pos_normals)
    
    # Y_negative mesh
    Y_neg_centroids = find_centroids(lattice= input_lattice_padded, ref_lattice=lat, dir= -1, axes= 1)
    Y_neg_mesh = construct_mesh_y_neg(Y_neg_centroids, unit)
    vertical_meshes.extend(Y_neg_mesh)

    # move the test points so they are not inside mesh edge
    Y_neg_text_point = Y_neg_centroids + (lat.unit/2 + 0.01) * [0,1,0]
    test_points.extend(Y_neg_text_point)

    # find normal of squares: take every other triangle normal
    Y_neg_mesh = tm.util.concatenate(Y_neg_mesh)
    Y_neg_normals = Y_neg_mesh.face_normals[::2]
    test_point_normals.extend(Y_neg_normals)

    # X_positive mesh
    X_pos_centroids = find_centroids(lattice= input_lattice_padded, ref_lattice=lat, dir= 1, axes= 0)
    X_pos_mesh = construct_mesh_x_pos(X_pos_centroids, unit)
    vertical_meshes.extend(X_pos_mesh)

    # move the test points so they are not inside mesh edge
    X_pos_text_point = X_pos_centroids + (lat.unit/2 + 0.01) * [-1,0,0]
    test_points.extend(X_pos_text_point)

    # find normal of squares: take every other triangle normal
    X_pos_mesh = tm.util.concatenate(X_pos_mesh)
    X_pos_normals = X_pos_mesh.face_normals[::2]
    test_point_normals.extend(X_pos_normals)

    # X_negative mesh
    X_neg_centroids = find_centroids(lattice= input_lattice_padded, ref_lattice=lat, dir= -1, axes= 0)
    X_neg_mesh = construct_mesh_x_neg(X_neg_centroids, unit)
    vertical_meshes.extend(X_neg_mesh)

    # move the test points so they are not inside mesh edge
    X_neg_text_point = X_neg_centroids + (lat.unit/2 + 0.01) * [1,0,0]
    test_points.extend(X_neg_text_point)

    # find normal of squares: take every other triangle normal
    X_neg_mesh = tm.util.concatenate(X_neg_mesh)
    X_neg_normals = X_neg_mesh.face_normals[::2]
    test_point_normals.extend(X_neg_normals)

    return vertical_meshes, test_points, test_point_normals

def construct_horizontal_mesh(lat,unit):
    horizontal_meshes = []
    test_points = []
    test_point_normals = []

    # padding to avoid the rolling issue
    input_lattice_padded = np.pad(lat, 1, mode='constant',constant_values=0)

    # Z_positive mesh
    Z_pos_centroids = find_centroids(lattice= input_lattice_padded, ref_lattice=lat, dir= -1, axes= 2)
    Z_pos_mesh = construct_mesh_z_pos(Z_pos_centroids, unit)
    horizontal_meshes.extend(Z_pos_mesh)

    # move the test points so they are not inside mesh edge
    Z_pos_text_point = Z_pos_centroids + (lat.unit/2 + 0.01) * [0,0,1]
    test_points.extend(Z_pos_text_point)

    # find normal of squares: take every other triangle normal
    Z_pos_mesh = tm.util.concatenate(Z_pos_mesh)
    Z_pos_normals = Z_pos_mesh.face_normals[::2]
    test_point_normals.extend(Z_pos_normals) 

    return horizontal_meshes, test_points, test_point_normals

# ------------------------------------------------------ #
#  
# OBJECTIVE FUNCTIONS
# 
# ------------------------------------------------------ #


def crit_1_PV(variables, ref_lattice, vector, magnitude, environment):
    # first we create the lattice for the current configuration
    vars = np.around(variables).astype(int) # I need this to not get errors with zero-size arrays
    lattice = reshape_and_store_to_lattice(vars, ref_lattice)

    # create vertical and horizontal test points, meshes, and normals
    horizontal_meshes, horizontal_test_points, horizontal_test_point_normals = construct_horizontal_mesh(lattice,lattice.unit)
    vertical_meshes, vertical_test_points, vertical_test_points_normals = construct_vertical_mesh(lattice, lattice.unit)

    # combine the meshes
    roof_mesh = tm.util.concatenate(horizontal_meshes)
    facade_mesh = tm.util.concatenate(vertical_meshes)
    building_mesh = tm.util.concatenate(roof_mesh,facade_mesh)
    combined_meshes = tm.util.concatenate(building_mesh, environment)

    # shoot towards the skydome points from all of the voxels
    ray_per_ctr = np.tile(vector, [len(horizontal_test_points),1]) # daylighting ray for each centroid
    ctr_per_ray = np.tile(horizontal_test_points, [1, len(vector)]).reshape(-1, 3) # daylighting centroid for each ray
    val_per_ray = np.tile(magnitude, [len(horizontal_test_points),1]) # solar ray intensity on horizontal surface per m2 for each test centroid    

    # USING EMBREE, ray tracing is much faster: any hit means ray is blocked, we don't need all hits. Hits that DO make it: x intensity
    ray_hit = combined_meshes.ray.intersects_any(ray_origins= ctr_per_ray, ray_directions= -ray_per_ctr)
    blocked_rays = np.logical_not(ray_hit)
    
    # values per ray
    values = np.multiply(lattice.unit[0]*lattice.unit[1] * val_per_ray.flatten(), blocked_rays) 

    # Wh received on roof over the year
    crit_1_PV_potential = np.sum(values,dtype=np.int64) # overflow with 32 bits

    # values mapped per mesh pair (values for each surface)
    crit_1_pervox = np.sum(values.reshape(len(horizontal_test_points),len(vector)),axis=1, dtype=np.int64) 

    return crit_1_PV_potential, crit_1_pervox


def crit_2_DL(variables, ref_lattice, vector, magnitude, environment):
    # first we create the lattice for the current configuration
    vars = np.around(variables).astype(int) # I need this to not get errors with zero-size arrays
    lattice = reshape_and_store_to_lattice(vars, ref_lattice)

    # create vertical and horizontal test points, meshes, and normals
    horizontal_meshes, horizontal_test_points, horizontal_test_point_normals = construct_horizontal_mesh(lattice,lattice.unit)
    vertical_meshes, vertical_test_points, vertical_test_points_normals = construct_vertical_mesh(lattice, lattice.unit)

    # combine the meshes
    roof_mesh = tm.util.concatenate(horizontal_meshes)
    facade_mesh = tm.util.concatenate(vertical_meshes)
    building_mesh = tm.util.concatenate(roof_mesh,facade_mesh)
    combined_meshes = tm.util.concatenate(building_mesh, environment)

    # shoot towards the skydome points from all of the voxels
    ray_per_ctr = np.tile(vector, [len(vertical_test_points),1]) # daylighting ray for each centroid
    ctr_per_ray = np.tile(vertical_test_points, [1, len(vector)]).reshape(-1, 3) # daylighting centroid for each ray
    val_per_ray = np.tile(magnitude, [len(vertical_test_points),1]) # solar ray intensity on horizontal surface per m2 for each test centroid    

    # USING EMBREE, ray tracing is much faster: any hit means ray is blocked, we don't need all hits. Hits that DO make it: x intensity
    ray_hit = combined_meshes.ray.intersects_any(ray_origins= ctr_per_ray, ray_directions= -ray_per_ctr)
    blocked_rays = np.logical_not(ray_hit)

    # values per ray in lux*100
    values = np.multiply(lattice.unit[0]*lattice.unit[1] * val_per_ray.flatten(), blocked_rays) 

    # lux*100 received over the year
    crit_2_DL_potential = np.sum(values,dtype=np.int64) 

    # values mapped per mesh pair (values for each surface)
    crit_2_pervox = np.sum(values.reshape(len(vertical_test_points),len(vector)),axis=1, dtype=np.int64) 

    return crit_2_DL_potential, crit_2_pervox


def crit_3_RC(variables, ref_lattice):
    # create the current configuration as a lattice
    vars = np.around(variables).astype(int) # I need this to not get errors with zero-size arrays
    curr_envelope = reshape_and_store_to_lattice(np.array(vars).astype('bool'), ref_lattice)

    # flatten the envelope
    envlp_voxs = curr_envelope.flatten()

    # create stencil
    stencil = tg.create_stencil("von_neumann", 1, 1)
    stencil.set_index([0,0,0], 0)

    # find indices of the neighbours for each voxel 
    neighs = curr_envelope.find_neighbours(stencil)

    # occupation status for the neighbours for each voxel
    neighs_status = envlp_voxs[neighs]

    # for voxels inside the envelope:
    neigh_array = np.array(neighs_status[envlp_voxs.astype("bool")])  

    # when the neighbour's status is False that refers to an outer face
    outer_faces = np.count_nonzero(neigh_array==0)

    # voxel edge length
    l = ref_lattice.unit[0] # TODO: can we leave this dimension out?

    # calculate total surface area of outer faces
    A_exterior = (l**2)*outer_faces

    # number of in-envelope voxels
    in_voxels = np.count_nonzero(variables)

    # calculate total volume inclosed in the envelope
    V = in_voxels * (l**3)

    # edge length of a cube that has the same volume
    l_ref = V**(1/3)

    # calculate ratio
    R_ref = (6*(l_ref**2))/V

    # dimensionless indicator heat retention potential
    crit_3_rc = (A_exterior/V)/R_ref
    
    return crit_3_rc

def crit_4_FSI(variables, ref_lattice, target):
    # calculate area of voxel
    vars = np.around(variables).astype(int) # I need this to not get errors with zero-size arrays
    vox_area = ref_lattice.unit[0] * ref_lattice.unit[1]

    # calculate area of the building plot
    site_area = ref_lattice.shape[0]*ref_lattice.shape[1] * vox_area

    # count number of active voxels
    vox_active = np.count_nonzero(vars)

    # calculate area of configuration
    config_area = vox_active * vox_area

    # calculate achieved FSI
    FSI = config_area / site_area

    # calculate difference between achieved FSI and target FSI. Equals 1 when target is reached, achieves diminishing returns afterwards
    crit_4_fsi = 2*FSI / (FSI + target)

    return crit_4_fsi