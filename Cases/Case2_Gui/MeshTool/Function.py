# All functions uesd in class 'meshTool' in 'MeshTool.py'
import numpy as np
import math
import matplotlib.pyplot as plt


def calDistance(Mesh_Firstlayer, Expansion_rate, MeshNum):
    '''
    Calculate the distance by three args.
    Args:
        Mesh_Firstlayer: The distance of the first mesh layer.
        Expansion_rate: Expansion rate of the mesh layers.
        MeshNum: Number of mesh layers.
    Returns:
        Distance.
    '''
    Distance = Mesh_Firstlayer * (1-Expansion_rate**MeshNum) / (1-Expansion_rate)
    return Distance


def calMeshNum(Distance, Mesh_Firstlayer, Expansion_rate):
    '''
    Calculate the exact and ceiling number of mesh layers based on three args.
    Formula used: Based on the geometric series sum formula: S = a1 * (1-r^n) / (1-r)
    '''
    MeshNum_exact = math.log( (1 - Distance * (1-Expansion_rate) / Mesh_Firstlayer), Expansion_rate)
    MeshNum_ceil = math.ceil(MeshNum_exact)
    return MeshNum_exact, MeshNum_ceil


def calExpansionRate(Distance, Mesh_Firstlayer, Expansion_rate):
    '''
    Calculate the actual expansion rate based on mesh lengh, first layer size and mesh number.
    The actual expansion rate will be slight smaller than the given expansion rate.
    Formula used: Based on the geometric series sum formula: S = a1 * (1-r^n) / (1-r)
    The polynomial equation to solve is: r^n - (S/a1)*r + (S/a1 - 1) = 0
    '''
    MeshNum = calMeshNum(Distance, Mesh_Firstlayer, Expansion_rate)[1] # Actual mesh number (ceiling value)
    # print(Distance, Mesh_Firstlayer, Expansion_rate, MeshNum)
    if MeshNum * Mesh_Firstlayer >= Distance:
        MeshNum = MeshNum - 1 # Adjust for small expansion rates and small distance
    coefficients = np.zeros(MeshNum+1) # Coefficients of the polynomial equation
    coefficients[0] = 1 # Coefficient of the highest degree term
    coefficients[-2] = -(Distance / Mesh_Firstlayer) # Coefficient of the linear term
    coefficients[-1] = (Distance / Mesh_Firstlayer) - 1 # Constant term
    # print(coefficients)
    roots = np.roots(coefficients) # Calculate the roots
    # print(roots)
    lower_bound = 1
    upper_bound = Expansion_rate + 5 # Define the range for the valid root
    filtered_roots = [root for root in roots if np.isreal(root) and lower_bound <= root <= upper_bound]
    ExpRate_rev = find_larger_real_part(filtered_roots) # Select the larger real part within the range
    return ExpRate_rev


def find_larger_real_part(complex_numbers):
    '''
    Find the largest real part.
    '''
    max_real = complex_numbers[0].real
    for num in complex_numbers:
        if num.real > max_real:
            max_real = num.real
    return max_real



def generate_coord(Feature_geometry):
    '''
    Generate unique coordinates for mesh generation based on the input feature geometry.

    This function takes the feature geometry as input, concatenates all coordinates into a single array,
    and extracts unique x, y, and z coordinates.

    Args:
        Feature_geometry (numpy.ndarray): Input feature geometry.
            Expected format: np.array([[x1min, y1min, z1min, x1max, y1max, z1max],
                              [x2min, y2min, z2min, x2max, y2max, z2max],
                              [x3min, y3min, z3min, x3max, y3max, z3max]])
            Each row represents a feature with start and end coordinates.

    Returns:
        tuple: A tuple containing the unique coordinates:
            - coordinate_x (numpy.ndarray): Unique x-coordinates.
            - coordinate_y (numpy.ndarray): Unique y-coordinates.
            - coordinate_z (numpy.ndarray): Unique z-coordinates.
    '''
    # Concatenate all coordinates into a single array
    all_coordinates = np.concatenate((Feature_geometry[:, :3], Feature_geometry[:, 3:]), axis=0)
    
    # Extract unique x, y, and z coordinates
    coordinate_x = np.unique(all_coordinates[:, 0], axis=0)
    coordinate_y = np.unique(all_coordinates[:, 1], axis=0)
    coordinate_z = np.unique(all_coordinates[:, 2], axis=0)
    return coordinate_x, coordinate_y, coordinate_z


def cal_pointIndex(coordinate_x, coordinate_y, coordinate_z):
    '''
    In OpenFOAM blockMesh, using 'count' directly to represent points is not intuitive. 
    Therefore, this function defines the point indices in advance in the blockMeshDict file. 
    When using these indices later, they can be referred to as $index_i_j_k, 
    which makes the blockMesh configuration more readable and easier to manage.
    Generate point indices in sequential order based on the given coordinates.
    '''
    count = 0
    index_array = np.zeros( (len(coordinate_x), len(coordinate_y), len(coordinate_z)) )
    for i in range(len(coordinate_x)):
        for j in range(len(coordinate_y)):
            for k in range(len(coordinate_z)):
                index_array[i, j, k] = count
                count += 1
    return index_array.astype(int)


def calMeshNumSimpleGrading(Mesh_Firstlayer, Mesh_Max, Expansion_rate, coordinate_x):
    '''
    In OpenFOAM blockMeshDict, a block should be defined as the form below:
    hex (0 170 187 17 1 171 188 18) (19 15 3) simpleGrading (1.0 1.0 1.0)
    simpleGrading only allows single-direction expansion.
    This fuction will divided the blocks into sub-block to achieve bi-directional expansion. 
    It dynamically calculates the number of sub-blocks, mesh cells, and expansion rates.
    

    Args:
        Mesh_Firstlayer: The size of the first layer of the mesh.
        Mesh_Max: The maximum size of the mesh cells.
        Expansion_rate: The expansion rate of the mesh cells.
        coordinate_x: The coordinates of the block boundaries in the x-direction.

    Returns:
        tuple: A tuple containing:
            - coordinate_x_exp (numpy.ndarray): The updated coordinates with intermediate points for expansion.
            - Mesh_x_exp (numpy.ndarray): The number of mesh cells for each block or sub-block.
            - SG_x_exp (numpy.ndarray): The expansion rates for each block or sub-block.
            - numAreas_x (numpy.ndarray): The number of sub-blocks for each block.
    '''
    distance_x = coordinate_x[1:] - coordinate_x[0:-1] # Length of each block
    numAreas_x = np.ones_like(distance_x).astype(int) # Number of sub-blocks for each block, intialized to 1
    coordinate_x_exp = coordinate_x # Initialize the expanded coordinates with the original coordinates
    Mesh_x_exp = np.array([]).astype(int)
    SG_x_exp = np.array([])
    
    # Calculate numAreas(number of sub-blocks)、MeshNum、simpleGrading
    for i in range(len(numAreas_x)):
        curDist = round(distance_x[i],6) # Length of the current block
        if curDist <= 3*Mesh_Firstlayer:
            numAreas_x[i] = 1 # one sub-block
            curMesh = math.ceil(curDist / Mesh_Firstlayer)
            curExp = 1.0
            Mesh_x_exp = np.append(Mesh_x_exp, [curMesh])
            SG_x_exp = np.append(SG_x_exp, [curExp**(curMesh-1)])
        elif curDist > 3*Mesh_Firstlayer and curDist <= (2+Expansion_rate)*Mesh_Firstlayer:
            numAreas_x[i] = 3 # three sub-block, one mesh for each sub-block
            curMesh_1 = 1
            curMesh_2 = 1
            curMesh_3 = 1
            curExp_1 = 1.0
            curExp_2 = 1.0
            curExp_3 = 1.0
            interCoordinate_1 = round(Mesh_Firstlayer + coordinate_x[i], 6)
            interCoordinate_2 = round(curDist - Mesh_Firstlayer + coordinate_x[i], 6)
            coordinate_x_exp = np.append(coordinate_x_exp, [interCoordinate_1, interCoordinate_2])
            Mesh_x_exp = np.append(Mesh_x_exp, [curMesh_1, curMesh_2, curMesh_3])
            SG_x_exp = np.append(SG_x_exp, [curExp_1**(curMesh_1-1), curExp_2**(curMesh_2-1), curExp_3**(curMesh_3-1)])
        elif curDist > (2+Expansion_rate)*Mesh_Firstlayer and curDist <= 4*Mesh_Firstlayer:
            numAreas_x[i] = 1 # one sub-block, using uniform mesh in this sub-block
            curMesh = math.ceil(curDist / Mesh_Firstlayer)
            curExp = 1.0
            Mesh_x_exp = np.append(Mesh_x_exp, [curMesh])
            SG_x_exp = np.append(SG_x_exp, [curExp**(curMesh-1)])
        else:
            # for longer blocks, determine the number of sub-blocks based on the maxium mesh size
            thersholdNum = math.log( Mesh_Max / Mesh_Firstlayer, Expansion_rate) + 1
            thersholdDist = 2 * calDistance(Mesh_Firstlayer, Expansion_rate, thersholdNum)
            if curDist < thersholdDist:
                numAreas_x[i] = 2 # the block length is less than the threshold distance, use two sub-blocks
                curMesh_1 = calMeshNum(curDist/2, Mesh_Firstlayer, Expansion_rate)[1]
                curMesh_2 = curMesh_1
                curExp_1 = calExpansionRate(curDist/2, Mesh_Firstlayer, Expansion_rate)
                curExp_2 = 1/calExpansionRate(curDist/2, Mesh_Firstlayer, Expansion_rate)
                interCoordinate_1 = round(curDist / 2 + coordinate_x[i], 6)
                coordinate_x_exp = np.append(coordinate_x_exp, [interCoordinate_1])
                Mesh_x_exp = np.append(Mesh_x_exp, [curMesh_1, curMesh_2])
                SG_x_exp = np.append(SG_x_exp, [round(curExp_1**(curMesh_1-1),6), round(curExp_2**(curMesh_2-1),6)])
            else:
                numAreas_x[i] = 3 # the block length is longer than the threshold distance, use three sub-blocks
                curMesh_2 = math.ceil( (curDist-thersholdDist) / Mesh_Max ) # num of mesh of the middle-sub-block
                midDist = curMesh_2 * Mesh_Max # the actural length of the middle-sub-block
                curMesh_1 = calMeshNum((curDist-midDist)/2, Mesh_Firstlayer, Expansion_rate)[1]
                curMesh_3 = curMesh_1
                curExp_1 = calExpansionRate((curDist-midDist)/2, Mesh_Firstlayer, Expansion_rate)
                curExp_2 = 1.0 # uniform mesh with max mesh size, in the middle-sub-block
                curExp_3 = 1/calExpansionRate((curDist-midDist)/2, Mesh_Firstlayer, Expansion_rate)
                interCoordinate_1  = round((curDist-midDist)/2 + coordinate_x[i], 6)
                interCoordinate_2  = round((curDist+midDist)/2 + coordinate_x[i], 6)
                coordinate_x_exp = np.append(coordinate_x_exp, [interCoordinate_1, interCoordinate_2])
                Mesh_x_exp = np.append(Mesh_x_exp, [curMesh_1, curMesh_2, curMesh_3])
                SG_x_exp = np.append(SG_x_exp, [round(curExp_1**(curMesh_1-1),6), curExp_2, round(curExp_3**(curMesh_3-1),6)])
    coordinate_x_exp.sort()
    return coordinate_x_exp, Mesh_x_exp, SG_x_exp, numAreas_x


def indexTransformer(numAreas_x, numAreas_y, numAreas_z, oldIndex):
    '''
    Transform block indices to sub-block indices.
    Args:
        numAreas_x (list or numpy.ndarray): Number of sub-blocks in each block in the x-direction.
        numAreas_y (list or numpy.ndarray): Number of sub-blocks in each block in the y-direction.
        numAreas_z (list or numpy.ndarray): Number of sub-blocks in each block in the z-direction.
        oldIndex (numpy.ndarray): The block indices to be transformed.
            - If a 1D array, it represents a boundary face.
            - If a 2D array, it represents 'obstacles' or 'restricted boundary face'.
    '''
    newIndex = np.zeros_like(oldIndex).astype(int)
    if oldIndex.size == 0:
        return oldIndex
    
    elif len(oldIndex.shape) == 1:
        newIndex[0] = sum(numAreas_x[0:oldIndex[0]])
        newIndex[1] = sum(numAreas_y[0:oldIndex[1]])
        newIndex[2] = sum(numAreas_z[0:oldIndex[2]])
        newIndex[3] = sum(numAreas_x[0:oldIndex[3]])
        newIndex[4] = sum(numAreas_y[0:oldIndex[4]])
        newIndex[5] = sum(numAreas_z[0:oldIndex[5]])
    else:
        for i in range(oldIndex.shape[0]):
            newIndex[i,0] = sum(numAreas_x[0:oldIndex[i,0]])
            newIndex[i,1] = sum(numAreas_y[0:oldIndex[i,1]])
            newIndex[i,2] = sum(numAreas_z[0:oldIndex[i,2]])
            newIndex[i,3] = sum(numAreas_x[0:oldIndex[i,3]])
            newIndex[i,4] = sum(numAreas_y[0:oldIndex[i,4]])
            newIndex[i,5] = sum(numAreas_z[0:oldIndex[i,5]])
    return newIndex


def generate_pointIndex(coordinate_x, coordinate_y, coordinate_z, output):
    '''
    Generate point indices in the same order as 'cal_pointIndex'
    Write them to a file (e.g.,blockMeshDict)
    '''
    count = 0
    with open(output, 'a') as f:
        f.write("\n")
        for i in range(len(coordinate_x)):
            for j in range(len(coordinate_y)):
                for k in range(len(coordinate_z)):
                    variable_name = f"index_{i}_{j}_{k}"
                    exec(f"{variable_name} = {count}")
                    f.write("\n")
                    f.write(f"{variable_name} {count};")
                    count += 1

                    
def generate_vertices(coordinate_x, coordinate_y, coordinate_z, output):
    '''
    Generate the 'vertices' section for blockMeshDict, 
    listing coordinates in the order defined by 'count'in 'generate_pointIndex'.
    
    Example is below:
    vertices
    (
        (0.0 0.0 0.0)		//0	index_0_0_0
        (0.0 0.0 0.15)		//1	index_0_0_1
        (0.0 0.0 0.55)		//2	index_0_0_2
        (0.0 0.0 0.75)		//3	index_0_0_3
    );
    '''
    count = 0
    with open(output, 'a') as f:
        f.write("\n\n")
        f.write(f"vertices")
        f.write("\n")
        f.write(f"(")
        for i in range(len(coordinate_x)):
            for j in range(len(coordinate_y)):
                for k in range(len(coordinate_z)):
                    x_i_j_k = round(coordinate_x[i], 6)
                    y_i_j_k = round(coordinate_y[j], 6)
                    z_i_j_k = round(coordinate_z[k], 6)
                    f.write("\n")
                    f.write(f"\t({x_i_j_k} {y_i_j_k} {z_i_j_k})\t\t//{count}\tindex_{i}_{j}_{k}")
                    count += 1
        f.write("\n")
        f.write(f");")


def generate_blocks(coordinate_x, coordinate_y, coordinate_z,
                    Mesh_x, Mesh_y, Mesh_z,
                    SimpleGrading_x, SimpleGrading_y,SimpleGrading_z,
                    output, 
                    solid_coordinate = np.array([])):
    '''
    Generate the 'blocks' section for blockMeshDict, 
    including optional solid regions marked with comments.
    
    Args:
        coordinate_x (list): List of x-coordinates.
        coordinate_y (list): List of y-coordinates.
        coordinate_z (list): List of z-coordinates.
        Mesh_x (list): Number of mesh cells in the x-direction for each block.
        Mesh_y (list): Number of mesh cells in the y-direction for each block.
        Mesh_z (list): Number of mesh cells in the z-direction for each block.
        SimpleGrading_x (list): Expansion ratio in the x-direction for each block.
        SimpleGrading_y (list): Expansion ratio in the y-direction for each block.
        SimpleGrading_z (list): Expansion ratio in the z-direction for each block.
        output (str): Path to the output file (e.g., blockMeshDict).
        solid_coordinate (numpy.ndarray, optional): Coordinates of solid regions.
            Format: np.array([[x1min, y1min, z1min, x1max, y1max, z1max], 
                        [x2min, y2min, z2min, x2max, y2max, z2max]]).
            Defaults to an empty array.
    
    Example is below:
    blocks
    (
        //hex (0 170 187 17 1 171 188 18) (19 15 3) simpleGrading (1.0 1.0 1.0)
        //hex (1 171 188 18 2 172 189 19) (19 15 8) simpleGrading (1.0 1.0 1.0)
        hex (2 172 189 19 3 173 190 20) (19 15 4) simpleGrading (1.0 1.0 1.0)
        hex (3 173 190 20 4 174 191 21) (19 15 2) simpleGrading (1.0 1.0 1.0)
        hex (4 174 191 21 5 175 192 22) (19 15 7) simpleGrading (1.0 1.0 1.0)
        hex (5 175 192 22 6 176 193 23) (19 15 3) simpleGrading (1.0 1.0 1.0)
        hex (6 176 193 23 7 177 194 24) (19 15 4) simpleGrading (1.0 1.0 1.0)
        hex (7 177 194 24 8 178 195 25) (19 15 2) simpleGrading (1.0 1.0 1.0)
    );
    '''
    index_array = cal_pointIndex(coordinate_x, coordinate_y, coordinate_z) # store the index for each point
    
    # Make an empty matrix to mark solid regions with comment symboks
    solid_matrix = np.empty((len(coordinate_x)-1, len(coordinate_y)-1, len(coordinate_z)-1), dtype=str)
    
    # Initialize a matrix element as empty strings
    for i in range(len(coordinate_x)-1):
        for j in range(len(coordinate_y)-1):
            for k in range(len(coordinate_z)-1):
                solid_matrix[i][j][k] = ''
                
    # Mark solid regions with comment symbols '//'
    for n in range(solid_coordinate.shape[0]):
        for i in range(solid_coordinate[n, 0],solid_coordinate[n, 3]):
            for j in range(solid_coordinate[n, 1],solid_coordinate[n, 4]):
                for k in range(solid_coordinate[n, 2],solid_coordinate[n, 5]):
                    solid_matrix[i][j][k] = '/'
    
    # Output the 'block' section in blockMeshDict, 
    with open(output, 'a') as f:
        f.write("\n\n")
        f.write(f"blocks")
        f.write("\n")
        f.write(f"(")
        for i in range( len(coordinate_x) - 1 ):
            for j in range( len(coordinate_y) - 1 ):
                for k in range( len(coordinate_z) - 1 ):
                    cube_index_0 = index_array[i  , j  , k  ]
                    cube_index_1 = index_array[i+1, j  , k  ]
                    cube_index_2 = index_array[i+1, j+1, k  ]
                    cube_index_3 = index_array[i  , j+1, k  ]
                    cube_index_4 = index_array[i  , j  , k+1]
                    cube_index_5 = index_array[i+1, j  , k+1]
                    cube_index_6 = index_array[i+1, j+1, k+1]
                    cube_index_7 = index_array[i  , j+1, k+1]
                    f.write("\n")
                    f.write(f"\t{solid_matrix[i][j][k]}{solid_matrix[i][j][k]}hex "
                            f"({cube_index_0} {cube_index_1} {cube_index_2} {cube_index_3} "
                            f"{cube_index_4} {cube_index_5} {cube_index_6} {cube_index_7}) "
                            f"({Mesh_x[i]} {Mesh_y[j]} {Mesh_z[k]}) "
                            f"simpleGrading ({SimpleGrading_x[i]} {SimpleGrading_y[j]} {SimpleGrading_z[k]})")
        f.write("\n")
        f.write(f");")


def generate_coordIndex(coordinate_x, coordinate_y, coordinate_z, coordAbs):
    '''
    Convert absolute coordinates of point into index of point.

    This function takes the unique coordinates and absolute coordinates as input,
    and returns the index of the absolute coordinates in the unique coordinate arrays.

    Args:
        coordinate_x (numpy.ndarray): Unique x-coordinates.
        coordinate_y (numpy.ndarray): Unique y-coordinates.
        coordinate_z (numpy.ndarray): Unique z-coordinates.
        coordAbs (numpy.ndarray): Absolute coordinates to be converted.
            - If a 1D array, it represents a single boundary face.
            - If a 2D array, it represents obstacle or restricted faces.

    Returns:
        numpy.ndarray: Indices of the absolute coordinates in the unique coordinate arrays.
    '''
    coordIndex = np.zeros_like(coordAbs).astype(int)
    if coordAbs.size == 0:
        return coordAbs

    elif len(coordAbs.shape) == 1: # If the input is a 1D array, it represents a single boundary face
        coordIndex[0] = np.where(coordinate_x == coordAbs[0])[0].item()
        coordIndex[1] = np.where(coordinate_y == coordAbs[1])[0].item()
        coordIndex[2] = np.where(coordinate_z == coordAbs[2])[0].item()
        coordIndex[3] = np.where(coordinate_x == coordAbs[3])[0].item()
        coordIndex[4] = np.where(coordinate_y == coordAbs[4])[0].item()
        coordIndex[5] = np.where(coordinate_z == coordAbs[5])[0].item()

    else:
        for i in range(coordAbs.shape[0]): # If the input is a 2D array, it represents obstacle or restricted faces
            coordIndex[i,0] = np.where(coordinate_x == coordAbs[i,0])[0].item()
            coordIndex[i,1] = np.where(coordinate_y == coordAbs[i,1])[0].item()
            coordIndex[i,2] = np.where(coordinate_z == coordAbs[i,2])[0].item()
            coordIndex[i,3] = np.where(coordinate_x == coordAbs[i,3])[0].item()
            coordIndex[i,4] = np.where(coordinate_y == coordAbs[i,4])[0].item()
            coordIndex[i,5] = np.where(coordinate_z == coordAbs[i,5])[0].item()
            
    return coordIndex


def generate_boundary_start(output):
    '''
    Write the start of the 'boundary' section in blockMeshDict.
    '''
    with open(output, 'a') as f:
        f.write("\n\n")
        f.write(f"boundary")
        f.write("\n")
        f.write(f"(")
def generate_boundary_end(output):
    '''
    Write the end of the 'boundary' section in blockMeshDict.
    '''
    with open(output, 'a') as f:
        f.write("\n")
        f.write(f");")
def generate_boundary_define_start(output, B_name, B_type):
    '''
    Start defining a boundary region in 'boundary' section
    '''
    with open(output, 'a') as f:
        f.write("\n")
        f.write(f"\t" + B_name)
        f.write("\n")
        f.write(f"\t" + "{")
        f.write("\n")
        f.write(f"\t\ttype " + B_type + ";")
        f.write("\n")
        f.write(f"\t\tfaces")
        f.write("\n")
        f.write(f"\t\t(")
def generate_boundary_define_end(output):
    '''
    End defining a boundary region in 'boundary' section
    '''
    with open(output, 'a') as f:
        f.write("\n")
        f.write(f"\t\t);")
        f.write("\n")
        f.write(f"\t" + "}")

        
def Judge_normal_vector(face_point = np.array([])):
    '''
    Determine the normal vector of a face based on the input 2 vertices.
    Args:
        face_point:
        A 1D array of shape (6,) containing the coordinates of the face vertices.
        Example: np.array([x1, y1, z1, x2, y2, z2])
    '''
    if np.array_equal(face_point[0], face_point[3]):
        return np.array([1, 0, 0])
    elif np.array_equal(face_point[1], face_point[4]):
        return np.array([0, 1, 0])
    elif np.array_equal(face_point[2], face_point[5]):
        return np.array([0, 0, 1])
    else:
        print(f"The input face is not an orthogonal face!")

        
def generate_face_index(numAreas_x, numAreas_y, numAreas_z, 
                        output, 
                        face_point, 
                        if_expansion, 
                        face_restricted_point = np.array([])):
    '''
    Generate face indices for the 'boundary' section in blockMeshDict.
    This function supports orthogonal faces.
    This function can handle restricted faces that need to be commented out.
    
    Ags:
        face_point: 
            A 1D array of shape (6,) containing the coordinates of the face.
            Format: np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        if_expansion (bool): 
            Whether to use index transformation for expansion meshes.
        face_restricted_point: 
            A 2D array containing the coordinates of restricted faces.
            Format: np.array([[x1min, y1min, z1min, x1max, y1max, z1max],
                        [x2min, y2min, z2min, x2max, y2max, z2max]])
            Defaults to an empty array.
   
   Example is below:
   boundary
    (
        upwall
        {
            type wall
            faces
            (
                (0 1 2 3)
            );
        }

        floorwall
        {
            type wall
            faces
            (
                (0 3 7 4)
                //(8 11 15 12)
            );
        }
    );
    '''
    # If expansion is uesd, func 'indexTransformer' will be used.
    if if_expansion:
        face_point = indexTransformer(numAreas_x, numAreas_y, numAreas_z, face_point)
        face_restricted_point = indexTransformer(numAreas_x, numAreas_y, numAreas_z, face_restricted_point)
    
    
    if np.array_equal(Judge_normal_vector(face_point), np.array([0, 0, 1])):
        # Face is in the xy-plane
        min_x = face_point[0]
        max_x = face_point[3]
        min_y = face_point[1]
        max_y = face_point[4]
        z = face_point[2]
        
        # Create a matrix to mark restricted faces
        face_restricted_matrix = np.empty((np.sum(numAreas_x), np.sum(numAreas_y)), dtype=str)
        for i in range(np.sum(numAreas_x)):
            for j in range(np.sum(numAreas_y)):
                face_restricted_matrix[i][j] = ''
        # Mark restricted faces with comment symbols "//"
        for n in range(face_restricted_point.shape[0]):
            min_x_r = face_restricted_point[n, 0]
            max_x_r = face_restricted_point[n, 3]
            min_y_r = face_restricted_point[n, 1]
            max_y_r = face_restricted_point[n, 4]
            for i in range(min_x_r, max_x_r):
                for j in range(min_y_r, max_y_r):
                    face_restricted_matrix[i][j]= '/'
        
        # Write the face indices to the output file
        with open(output, 'a') as f:
            for i in range( min_x, max_x ):
                for j in range( min_y, max_y ):
                    for k in range( z, z+1 ):
                        f.write("\n")
                        f.write(f"\t\t\t" + 
                              f"{face_restricted_matrix[i][j]}{face_restricted_matrix[i][j]}" + 
                              f"(" + 
                              f"$index_{i}_{j}_{k} $index_{i}_{j+1}_{k} $index_{i+1}_{j+1}_{k} $index_{i+1}_{j}_{k}" + 
                              f")")
                           
    elif np.array_equal(Judge_normal_vector(face_point), np.array([0, 1, 0])):
        # Face is in the xz-plane
        min_x = face_point[0]
        max_x = face_point[3]
        min_z = face_point[2]
        max_z = face_point[5]
        y = face_point[1]
        
        # Create a matrix to mark restricted faces
        face_restricted_matrix = np.empty((np.sum(numAreas_x), np.sum(numAreas_z)), dtype=str)
        for i in range(np.sum(numAreas_x)):
            for k in range(np.sum(numAreas_z)):
                face_restricted_matrix[i][k] = ''
        # Mark restricted faces with comment symbols "//"
        for n in range(face_restricted_point.shape[0]):
            min_x_r = face_restricted_point[n, 0]
            max_x_r = face_restricted_point[n, 3]
            min_z_r = face_restricted_point[n, 2]
            max_z_r = face_restricted_point[n, 5]
            for i in range(min_x_r, max_x_r):
                for k in range(min_z_r, max_z_r):
                    face_restricted_matrix[i][k]= '/'

        # Write the face indices to the output file
        with open(output, 'a') as f:
            for i in range( min_x, max_x ):
                for j in range( y, y+1 ):
                    for k in range( min_z, max_z ):
                        f.write("\n")
                        f.write(f"\t\t\t" + 
                              f"{face_restricted_matrix[i][k]}{face_restricted_matrix[i][k]}" + 
                              f"(" + 
                              f"$index_{i}_{j}_{k} $index_{i}_{j}_{k+1} $index_{i+1}_{j}_{k+1} $index_{i+1}_{j}_{k}" + 
                              f")")
                           
    elif np.array_equal(Judge_normal_vector(face_point), np.array([1, 0, 0])):
        # Face is in the yz-plane
        min_y = face_point[1]
        max_y = face_point[4]
        min_z = face_point[2]
        max_z = face_point[5]
        x = face_point[0]
        
        # Create a matrix to mark restricted faces
        face_restricted_matrix = np.empty((np.sum(numAreas_y), np.sum(numAreas_z)), dtype=str)
        for j in range(np.sum(numAreas_y)):
            for k in range(np.sum(numAreas_z)):
                face_restricted_matrix[j][k] = ''
        # Mark restricted faces with comment symbols "//"
        for n in range(face_restricted_point.shape[0]):
            min_y_r = face_restricted_point[n, 1]
            max_y_r = face_restricted_point[n, 4]
            min_z_r = face_restricted_point[n, 2]
            max_z_r = face_restricted_point[n, 5]
            for j in range(min_y_r, max_y_r):
                for k in range(min_z_r, max_z_r):
                    face_restricted_matrix[j][k]= '/'

        # Write the face indices to the output file
        with open(output, 'a') as f:
            for i in range( x, x+1 ):
                for j in range( min_y, max_y ):
                    for k in range( min_z, max_z ):
                        f.write("\n")
                        f.write(f"\t\t\t" + 
                              f"{face_restricted_matrix[j][k]}{face_restricted_matrix[j][k]}" + 
                              f"(" + 
                              f"$index_{i}_{j}_{k} $index_{i}_{j}_{k+1} $index_{i}_{j+1}_{k+1} $index_{i}_{j+1}_{k}" + 
                              f")")
    else:
        print("Unsupported face normal vector!")


def boundCheckTool_block(coordinate_x,coordinate_y,coordinate_z,solid_coordinate = np.array([])):
    '''
    Check boundary parts.
    The principle of this function is based on the method described in the following paper.
    'Development of a scripting tool for the fast and batch generation of orthogonal hexahedral mesh for CFD analysis in Built Environments'
    '''
    # Initialize boundaries
    boundCheck_x = np.zeros([len(coordinate_x)  ,len(coordinate_y)-1, len(coordinate_z)-1]).astype(int)
    boundCheck_y = np.zeros([len(coordinate_x)-1,len(coordinate_y)  , len(coordinate_z)-1]).astype(int)
    boundCheck_z = np.zeros([len(coordinate_x)-1,len(coordinate_y)-1, len(coordinate_z)  ]).astype(int)
    
    # Initialize block existence with 1 and 0, fluid is 1, solid is 0
    blockExist = np.ones((len(coordinate_x)-1, len(coordinate_y)-1, len(coordinate_z)-1), dtype=int)
    for n in range(solid_coordinate.shape[0]):
        for i in range(solid_coordinate[n, 0],solid_coordinate[n, 3]):
            for j in range(solid_coordinate[n, 1],solid_coordinate[n, 4]):
                for k in range(solid_coordinate[n, 2],solid_coordinate[n, 5]):
                    blockExist[i][j][k] = 0
                    
    # Update boundCheck which will be used to check boundary parts. 
    boundCheck_x[:-1, :, :] += blockExist
    boundCheck_x[ 1:, :, :] += blockExist
    boundCheck_y[:, :-1, :] += blockExist
    boundCheck_y[:,  1:, :] += blockExist
    boundCheck_z[:, :, :-1] += blockExist
    boundCheck_z[:, :,  1:] += blockExist
    
    return boundCheck_x, boundCheck_y, boundCheck_z


def boundCheckTool_boundary(boundCheck_x, boundCheck_y, boundCheck_z, 
                            coordinate_x,coordinate_y,coordinate_z, 
                            face_point, face_restricted_point = np.array([])):
    ''''
    Update boundary check tensors based on face definitions and restricted areas.

    This function updates the boundary check tensors (boundCheck_x, boundCheck_y, boundCheck_z) based on the given face and restricted faces.
    The principle of this function is based on the method described in the following paper.
    'Development of a scripting tool for the fast and batch generation of orthogonal hexahedral mesh for CFD analysis in Built Environments'
    Args:
        boundCheck_x (numpy.ndarray): Boundary check tensor for the x-direction.
        boundCheck_y (numpy.ndarray): Boundary check tensor for the y-direction.
        boundCheck_z (numpy.ndarray): Boundary check tensor for the z-direction.
        face_point (numpy.ndarray): Coordinates of the face.
            Format: np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        face_restricted_point (numpy.ndarray, optional): Coordinates of restricted faces.
            Format: np.array([[x1min, y1min, z1min, x1max, y1max, z1max],
                        [x2min, y2min, z2min, x2max, y2max, z2max]])
            Defaults to an empty array.
    '''
    if np.array_equal(Judge_normal_vector(face_point), np.array([0, 0, 1])):
        # Face is in the xy-plane
        z = face_point[2]
        # Define face-existence tensor
        faceExist = np.zeros([len(coordinate_x)-1, len(coordinate_y)-1, len(coordinate_z)]).astype(int)
        min_x = face_point[0]
        max_x = face_point[3]
        min_y = face_point[1]
        max_y = face_point[4]
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                faceExist[i][j][z] = 1
         # Set restricted faces to 0
        for n in range(face_restricted_point.shape[0]):
            min_x_r = face_restricted_point[n, 0]
            max_x_r = face_restricted_point[n, 3]
            min_y_r = face_restricted_point[n, 1]
            max_y_r = face_restricted_point[n, 4]
            for i in range(min_x_r, max_x_r):
                for j in range(min_y_r, max_y_r):
                    faceExist[i][j][z] = 0
        boundCheck_z += faceExist
                             
    elif np.array_equal(Judge_normal_vector(face_point), np.array([0, 1, 0])):
        # Face is in the xz-plane
        y = face_point[1]
        # Define face-existence tensor
        faceExist = np.zeros([len(coordinate_x)-1, len(coordinate_y), len(coordinate_z)-1]).astype(int)
        min_x = face_point[0]
        max_x = face_point[3]
        min_z = face_point[2]
        max_z = face_point[5]
        for i in range(min_x, max_x):
            for k in range(min_z, max_z):
                faceExist[i][y][k] = 1
        # Set restricted faces to 0
        for n in range(face_restricted_point.shape[0]):
            min_x_r = face_restricted_point[n, 0]
            max_x_r = face_restricted_point[n, 3]
            min_z_r = face_restricted_point[n, 2]
            max_z_r = face_restricted_point[n, 5]
            for i in range(min_x_r, max_x_r):
                for k in range(min_z_r, max_z_r):
                    faceExist[i][y][k] = 0
        boundCheck_y += faceExist
         
    elif np.array_equal(Judge_normal_vector(face_point), np.array([1, 0, 0])):
        # Face is in the yz-plane
        x = face_point[0]
        # Define face-existence tensor
        faceExist = np.zeros([len(coordinate_x), len(coordinate_y)-1, len(coordinate_z)-1]).astype(int)
        min_y = face_point[1]
        max_y = face_point[4]
        min_z = face_point[2]
        max_z = face_point[5]
        for j in range(min_y, max_y):
            for k in range(min_z, max_z):
                faceExist[x][j][k] = 1
        # Set restricted faces to 0
        for n in range(face_restricted_point.shape[0]):
            min_y_r = face_restricted_point[n, 1]
            max_y_r = face_restricted_point[n, 4]
            min_z_r = face_restricted_point[n, 2]
            max_z_r = face_restricted_point[n, 5]
            for j in range(min_y_r, max_y_r):
                for k in range(min_z_r, max_z_r):
                    faceExist[x][j][k] = 0
        boundCheck_x += faceExist
                    
    else:
        print("Unsupported face normal vector!")
        
    return boundCheck_x, boundCheck_y, boundCheck_z

def boundCheckTool_output(bCh_x_ref, boundCheck_x, if_end):
    '''
    Check the boundary partititoning and output detailed error messages.
    This function compares the reference boundary check tensor (bCh_x_ref) with the actual boundary check tensor (boundCheck_x)
    to identify potential issues in the boundary parts.
    The principle of this function is based on the method described in the following paper.
    'Development of a scripting tool for the fast and batch generation of orthogonal hexahedral mesh for CFD analysis in Built Environments'
    
    Args:
        bCh_x_ref (numpy.ndarray): Reference boundary check tensor.
        boundCheck_x (numpy.ndarray): Actual boundary check tensor.
        if_end (bool): Flag indicating whether this is the final check (True) or an intermediate check (False).
    '''
    bCh_x_bool = ((bCh_x_ref==0)&(boundCheck_x==0)) + ((bCh_x_ref==1)&(boundCheck_x==2)) + ((bCh_x_ref==2)&(boundCheck_x==2))
    if np.all(bCh_x_bool):
        print("Boundary partitioning is correct")
    else:
        '''
        Possible boundary partitioning issues:
        1. ref is 0, but the corresponding value is greater than 0: This face is an internal face in solid, cannot be partitioned boundary.
        2. ref is 1, but the corresponding value is 1: This face is not partitioned boundary.
        3. ref is 1, but the corresponding value is greater than 2: This face is partitioned boundary multiple times.
        4. ref is 2, but the corresponding value is greater than 2: This face is an internal face in fluid but is partitioned boundary.
        '''
        error_found = False        
        if np.any((bCh_x_ref==0)&(boundCheck_x>0)):
            wrong_positions = np.where((bCh_x_ref==0)&(boundCheck_x>0))
            print("Boundary partitioning issue detected!")
            print("Error positions:", wrong_positions)
            print("Error reason: This face is an internal face in solid, cannot be partitioned boundary！")
            error_found = True        
        if np.any((bCh_x_ref==1)&(boundCheck_x>2)):
            wrong_positions = np.where((bCh_x_ref==1)&(boundCheck_x>2))
            print("Boundary partitioning issue detected!")
            print("Error positions:", wrong_positions)
            print("Error reason: This face is partitioned boundary multiple times！")
            error_found = True            
        if np.any((bCh_x_ref==2)&(boundCheck_x>2)):
            wrong_positions = np.where((bCh_x_ref==2)&(boundCheck_x>2))
            print("Boundary partitioning issue detected!")
            print("Error positions:：", wrong_positions)
            print("Error reason: This face is an internal face in fluid, cannot be partitioned boundary！")
            error_found = True
        if np.any((bCh_x_ref==1)&(boundCheck_x==1)): # Only report this error at the end
            if if_end:
                wrong_positions = np.where((bCh_x_ref==1)&(boundCheck_x==1))
                print("Boundary partitioning issue detected!")
                print("Error positions:", wrong_positions)
                print("Error reason: This face need to be partitioned boundary！")
                error_found = True     
            else:
                print("No issues detected so far, but some faces still need boundary partitioning.")
                error_found = True
        if not error_found:
            print("Boundary partitioning issue detected!")
            print("Error reason: Unknown!")

            