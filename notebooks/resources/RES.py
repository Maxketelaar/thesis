import numpy as np
import scipy as sp
import topogenesis as tg
import pyvista as pv

# ------------------------------------------------------ #
# convert trimesh object to pyvista object
# from: SAzadadi
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