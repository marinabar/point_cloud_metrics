import trimesh

def add_jitter(mesh, jitter=0.01):
    noisy_mesh = trimesh.permutate.noise(mesh, magnitude=jitter)
    return noisy_mesh

def mirror_mesh_x(mesh):
    if mesh is None:
        return None
    
    point = [0, 0, 0]
    normal = [1, 0, 0]
    
    reflection_matrix = trimesh.transformations.reflection_matrix(point, normal)
    
    return mesh.apply_transform(reflection_matrix)

def rotate_mesh(mesh, angle, axis=[1, 0, 0], point=[0, 0, 0]):
    if mesh is None:
        return None
    
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis, point)
    transformed_mesh = mesh.apply_transform(rotation_matrix)

    return transformed_mesh
