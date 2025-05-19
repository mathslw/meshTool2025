import numpy as np
import math
import matplotlib.pyplot as plt
import inspect
import os
import shutil
from MeshTool.Function import *


class meshTool:
    def __init__(self, of_version, path, Feature_geometry, coordAbs, if_expansion, if_prBlock):
        self.path = path
        self.output = f'{path}/blockMeshDict'
        self.Feature_geometry = Feature_geometry
        self.coordAbs = coordAbs
        self.coordinate_x, self.coordinate_y, self.coordinate_z = generate_coord(Feature_geometry)
        self.distance_x = self.coordinate_x[1:] - self.coordinate_x[0:-1]
        self.distance_y = self.coordinate_y[1:] - self.coordinate_y[0:-1]
        self.distance_z = self.coordinate_z[1:] - self.coordinate_z[0:-1]
        self.solid_coordinate = generate_coordIndex(
            self.coordinate_x, self.coordinate_y, self.coordinate_z, self.coordAbs)
        self.bCh_x_ref, self.bCh_y_ref, self.bCh_z_ref = (
            boundCheckTool_block(self.coordinate_x,self.coordinate_y,self.coordinate_z,self.solid_coordinate)) 
        self.if_expansion = if_expansion
        self.boundaryDict = {}
        if if_prBlock:
            self.prBlockPartition()
        sourcePath = 'MeshTool/blockMeshAllVersions'
        self.supportedVersions = [f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath, f))]
        self.readOpenFoamVersion(of_version)
    
    
    def readOpenFoamVersion(self, of_version):
        source_file = f'MeshTool/blockMeshAllVersions/{of_version}/blockMeshDict'
        destination_folder = self.path

        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")

        # 检查源文件是否存在
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, destination_folder)
                print(f"Successfully copied '{of_version}' blockMeshDict to {destination_folder}")
            except Exception as e:
                print(f"Error occurred while copying: {e}")
        else:
            print(f"Unsupported version '{of_version}'，Supported versions are: {self.supportedVersions}")
        
    
    def prBlockPartition(self):
        print(f"Dividing the entire space into blocks based on geometric points: ")
        print(f"\tCoordinates in the x-direction: {self.coordinate_x}, with {len(self.coordinate_x)} points.")
        print(f"\tCoordinates in the y-direction: {self.coordinate_y}，with {len(self.coordinate_y)} points.")
        print(f"\tCoordinates in the z-direction: {self.coordinate_z}，with {len(self.coordinate_z)} points.")
        print(f"\tThe entire computational domain is divided into: {len(self.coordinate_x)-1} * {len(self.coordinate_y)-1} * {len(self.coordinate_z)-1} blocks\n")
        
        
    def judgeExpansion(self, interaction=True, params=None):
        if self.if_expansion:
            if interaction:
                print(f"You will use non-uniform meshes.")
                print(f"You need to set the following three parameters:")
                print(f"first layer mesh size, maximum mesh size, mesh growth rate, in the following format:")
                print(f"0.01,0.2,2")
                user_input = input("Please enter the parameters in order, seperate by ',':")
                try:
                    self.Mesh_Firstlayer, self.Mesh_Max, self.Expansion_rate = map(float, user_input.split(','))
                except ValueError:
                    print("Incorrect input format. Please try again.")
                    self.judgeExpansion()
            else:
                if params is None or len(params) != 3:
                    raise ValueError(
                        "In non-interactive mode, you must provide three parameters as a list:\n"
                        "    - first layer mesh size, maximum mesh size, mesh growth rate.\n"
                        "Example: [0.01, 0.2, 2]")
                self.Mesh_Firstlayer, self.Mesh_Max, self.Expansion_rate = map(float, params)
        else:
            if interaction:
                print(f"You will use uniform meshes.")
                print(f"You need to set the mesh size, for example:")
                print(f"0.1")
                user_input = input("Please enter the parameter: ")
                try:
                    self.Mesh_scale = float(user_input)
                except ValueError:
                    print("Incorrect input format. Please try again.")
                    self.judgeExpansion()
            else:
                if params is None or len(params) != 1:
                    raise ValueError(
                        "In non-interactive mode, you must provide one parameter as a list:\n"
                        "    - Mesh size\n"
                        "Example: [0.1]")
                self.Mesh_scale = float(params[0])
                
                
    def generateGrid(self, if_print):
        ''''
        Galculate the parameters of the 'block' section in blockMeshDict.
        Non-uniform or uniform meshes.
        '''
        if self.if_expansion:
            self.Mesh_Firstlayer += 0.0000001 # add a small value to avoid numerical issues
            self.coordinate_x_exp, self.Mesh_x_exp, self.SG_x_exp, self.numAreas_x = calMeshNumSimpleGrading(
                self.Mesh_Firstlayer, self.Mesh_Max, self.Expansion_rate, self.coordinate_x)
            self.coordinate_y_exp, self.Mesh_y_exp, self.SG_y_exp, self.numAreas_y = calMeshNumSimpleGrading(
                self.Mesh_Firstlayer, self.Mesh_Max, self.Expansion_rate, self.coordinate_y)
            self.coordinate_z_exp, self.Mesh_z_exp, self.SG_z_exp, self.numAreas_z = calMeshNumSimpleGrading(
                self.Mesh_Firstlayer, self.Mesh_Max, self.Expansion_rate, self.coordinate_z)
            self.distance_x_exp = self.coordinate_x_exp[1:] - self.coordinate_x_exp[0:-1]
            self.distance_y_exp = self.coordinate_y_exp[1:] - self.coordinate_y_exp[0:-1]
            self.distance_z_exp = self.coordinate_z_exp[1:] - self.coordinate_z_exp[0:-1]
        else:
            self.Mesh_scale += 0.0000001 # add a small value to avoid numerical issues
            self.Mesh_x = np.ceil( self.distance_x / self.Mesh_scale).astype(int)
            self.Mesh_y = np.ceil( self.distance_y / self.Mesh_scale).astype(int)
            self.Mesh_z = np.ceil( self.distance_z / self.Mesh_scale).astype(int)
            self.SimpleGrading_x = np.ones_like(self.distance_x)
            self.SimpleGrading_y = np.ones_like(self.distance_y)
            self.SimpleGrading_z = np.ones_like(self.distance_z)
            self.numAreas_x = np.ones_like(self.distance_x).astype(int)
            self.numAreas_y = np.ones_like(self.distance_y).astype(int)
            self.numAreas_z = np.ones_like(self.distance_z).astype(int)
        if if_print:
            self.prGrid()
            
            
    def prGrid(self):
        '''
        Print the mesh parameters for either non-uniform or uniform meshes.
        '''
        if self.if_expansion:
            print(f"Non-uniform mesh parameters are as follows:")
            print(f"\tNumber of sub-block partitions in the x-direction: {self.numAreas_x}")
            print(f"\tNumber of sub-block partitions in the y-direction:{self.numAreas_y}")
            print(f"\tNumber of sub-block partitions in the z-direction:{self.numAreas_z}")
            print(f"\tCoordinates in the x-direction: {self.coordinate_x_exp}, "+
                  f"with {len(self.coordinate_x_exp)} points")
            print(f"\tCoordinates in the y-direction: {self.coordinate_y_exp}, "+
                  f"with {len(self.coordinate_y_exp)} points")
            print(f"\tCoordinates in the z-direction: {self.coordinate_z_exp}, "+
                  f"with {len(self.coordinate_z_exp)} points")
            print(f"\tThe entire space is divided into "+
                  f"{len(self.coordinate_x_exp)-1} * "+
                  f"{len(self.coordinate_y_exp)-1} * "+
                  f"{len(self.coordinate_z_exp)-1} blocks")
            print(f"\tNumber of meshes in the x-direction: {self.Mesh_x_exp}")
            print(f"\tNumber of meshes in the y-direction: {self.Mesh_y_exp}")
            print(f"\tNumber of meshes in the z-direction: {self.Mesh_z_exp}")
            print(f"\tMesh growth rate in the x_direction: {np.round(self.SG_x_exp, 4)}")
            print(f"\tMesh growth rate in the y_direction: {np.round(self.SG_y_exp, 4)}")
            print(f"\tMesh growth rate in the z_direction: {np.round(self.SG_z_exp, 4)}")
            print(f"\tTotal mesh number: "+
                  f"{np.sum(self.Mesh_x_exp)} * {np.sum(self.Mesh_y_exp)} * {np.sum(self.Mesh_z_exp)}")
        else:
            print(f"Uniform mesh parameters are as follows:")
            print(f"\tNumber of meshes in the x-direction: {self.Mesh_x}")
            print(f"\tNumber of meshes in the y-direction: {self.Mesh_y}")
            print(f"\tNumber of meshes in the z-direction: {self.Mesh_z}")
            print(f"\tMesh growth rate in the x_direction: {self.SimpleGrading_x}")
            print(f"\tMesh growth rate in the y_direction: {self.SimpleGrading_y}")
            print(f"\tMesh growth rate in the z_direction: {self.SimpleGrading_z}")
            print(f"\tTotal mesh number: "+
                  f"{np.sum(self.Mesh_x)} * {np.sum(self.Mesh_y)} * {np.sum(self.Mesh_z)}")    
            
            
    def outputGrid(self):
        '''
        Output the point index, vertices section, blocks section to blockMeshDict.
        '''
        if self.if_expansion:
            self.solid_coordinate_exp = indexTransformer(
                self.numAreas_x, self.numAreas_y, self.numAreas_z, self.solid_coordinate)
            self.index_array = generate_pointIndex(
                self.coordinate_x_exp, self.coordinate_y_exp, self.coordinate_z_exp, self.output)
            print(f"Point indices successfully imported to --{self.output}--folder")
            generate_vertices(self.coordinate_x_exp, self.coordinate_y_exp, self.coordinate_z_exp, self.output)
            print(f"Point coordinates successfully imported to --{self.output}--folder")
            generate_blocks(self.coordinate_x_exp, self.coordinate_y_exp, self.coordinate_z_exp, 
                            self.Mesh_x_exp, self.Mesh_y_exp, self.Mesh_z_exp, 
                            self.SG_x_exp, self.SG_y_exp, self.SG_z_exp,
                            self.output, self.solid_coordinate_exp)
            print(f"Blocks and corresponding parameters successfully imported to --{self.output}--folder")
        else:
            self.index_array = generate_pointIndex(self.coordinate_x, self.coordinate_y, self.coordinate_z, self.output)
            print(f"Point indices successfully imported to --{self.output}--folder")
            generate_vertices(self.coordinate_x, self.coordinate_y, self.coordinate_z, self.output)
            print(f"Point coordinates successfully imported to --{self.output}--folder")
            generate_blocks(self.coordinate_x, self.coordinate_y, self.coordinate_z, 
                            self.Mesh_x, self.Mesh_y, self.Mesh_z, 
                            self.SimpleGrading_x, self.SimpleGrading_y, self.SimpleGrading_z,
                            self.output, self.solid_coordinate)
            print(f"Blocks and corresponding parameters successfully imported to --{self.output}--folder")
            
            
    def defineBoundaryCondition(self, interaction=True, params=None):
        if interaction:
            self.defineBoundaryCondition_interaction()
        else:
            self.defineBoundaryCondition_mamual(params)

            
    def defineBoundaryCondition_interaction(self):
        '''
         Interactively define boundary conditions and store them in self.boundaryDict.
        '''
        while True:
            # Define the boundary name.
            key_bName = input("Please enter the boundary name (e.g., 'inlet') or 'q' to quit")
            if key_bName.lower() == 'q':
                print("Exited boundary partitioning.")
                break
            if not isinstance(key_bName, str):
                print("Error: The boundary name must be a string.")
                continue
            self.boundaryDict[key_bName] = {} # Create a nested dictionary for the boundary
                
            # Define the boundary type.
            value_bType = input(f"Please enter the type for boundary '{key_bName}': ")
            if not isinstance(key_bName, str):
                print("Error: The boundary type must be a string.")
                continue
            self.boundaryDict[key_bName]['type'] = value_bType

            # Define the number of faces for this boundary.
            while True:
                face_nums = input(f"Please enter the number of faces for boundary '{key_bName}':" )
                try:
                    face_nums = int(face_nums)
                    self.boundaryDict[key_bName]['faceNum'] = face_nums
                    break
                except Exception as e:
                        print(f"Error: Unable to convert input to int. Exception: {e}")
                        print(f"Please enter a valid number")
                        continue
            
            # Define each face for this boundary.
            for face_num in range(1,face_nums+1):
                # whole face
                while True:
                    value_bFace = input(f"Please enter the diagonal coordinates for face {face_num} of boundary '{key_bName}': ")
                    try:
                        value_bFace = np.array(eval(value_bFace))
                        # check if 'value_bFace' is a 1D tensor
                        if not (isinstance(value_bFace, np.ndarray) and value_bFace.ndim == 1):
                            print("Error: This is a whole face, the input should be a 1D tensor!")
                            continue
                        self.boundaryDict[key_bName][f'face_{face_num}'] = value_bFace
                        break
                    except Exception as e:
                        print(f"Error: Unable to convert input to a NumPy array. Exception: {e}")
                        print(f"Please enter valid diagonal coordinates")
                        continue
                # restricted face
                while True:
                    value_bFaceRestricted = input(
                        f"Please enter the diagonal coordinates for the restricted faces of face {face_num} of boundary '{key_bName}', "+
                        f"format: [[x1min,y1min,z1min,x1max,y1max,z1max],[x2min,y2min,z2min,x2max,y2max,z2max]], or 'none' if none:")
                    if value_bFaceRestricted.lower() == "none":
                        self.boundaryDict[key_bName][f'faceRestricted_{face_num}'] = np.array([])
                        print("Detected 'none' input, no face to remove.")
                        break
                    try:
                        value_bFaceRestricted = np.array(eval(value_bFaceRestricted))
                        # check if 'value_bFaceRestricted' is a 2D tensor
                        if not (isinstance(value_bFaceRestricted, np.ndarray) and value_bFaceRestricted.ndim == 2):
                            print("Error: The restricted faces should be a two-dimensional NumPy tensor.")
                            continue
                        self.boundaryDict[key_bName][f'faceRestricted_{face_num}'] = value_bFaceRestricted
                        break
                    except Exception as e:
                        print(f"Error: Unable to convert input to a NumPy array. Exception: {e}")
                        print(f"Please enter valid diagonal coordinates")
                        continue
                        
            # After defining each boundary, automatically check the boundary conditions.
            # During the interaction process, it is allowed to have undefined boundary conditions!
            self.checkBoundaryCondition(if_end = False)
    
    
    def defineBoundaryCondition_mamual(self, params):
        self.boundaryDict = params
        print(f"Defined boundaries have been input into the tool.")
                
                
    def checkBoundaryCondition(self, if_end):
        '''
        Check the boundary partitioning for correctness.
        This function compares the reference boundary check tensors with the updated boundary check tensors
        to indentify potential issues in the boundary conditions.
        It can be used at any time during the interactive process or as a final check.
        '''
        # Copy the initial reference tensors.
        boundCheck_x = self.bCh_x_ref.copy()
        boundCheck_y = self.bCh_y_ref.copy()
        boundCheck_z = self.bCh_z_ref.copy()
        # Update the reference tensors based on 'boundaryDict'.
        for key in self.boundaryDict:
            B_name = key
            B_type = self.boundaryDict[B_name]['type']
            for i in range(1, self.boundaryDict[B_name]['faceNum']+1):
                face_point = self.boundaryDict[B_name][f'face_{i}']
                face_restricted_point = self.boundaryDict[B_name][f'faceRestricted_{i}']
                face_point = generate_coordIndex(self.coordinate_x, self.coordinate_y, self.coordinate_z, face_point)
                face_restricted_point = generate_coordIndex(self.coordinate_x, self.coordinate_y, self.coordinate_z, face_restricted_point)
                boundCheck_x, boundCheck_y, boundCheck_z = boundCheckTool_boundary(
                    boundCheck_x, boundCheck_y, boundCheck_z, 
                    self.coordinate_x, self.coordinate_y, self.coordinate_z,
                    face_point, face_restricted_point)

        # Check boundaries in the yz-plane
        print("Checking boundaries in the yz-plane: ")
        boundCheckTool_output(self.bCh_x_ref, boundCheck_x, if_end)
        # Check boundaries in the xz-plane
        print("Checking boundaries in the xz-plane: ")
        boundCheckTool_output(self.bCh_y_ref, boundCheck_y, if_end)
        # Check boundaries in the xy-plane
        print("Checking boundaries in the xy-plane: ")
        boundCheckTool_output(self.bCh_z_ref, boundCheck_z, if_end)          

    
    def delBoundaryCondition(self, key_bName):
        if key_bName in self.boundaryDict:
            del self.boundaryDict[key_bName]
            print(f"Boundary '{key_bName}' has been deleted.")
        else:
            print(f"No boundary named '{key_bName}' was found.")
        
    
    def prBoundaryCondition(self):
        for Key, Value in self.boundaryDict.items():
            print(f"-Boundary name: '{Key}'")
            print(f"---Boundary information: ")
            for key, value in Value.items():
                print(f"-----'{key}'")
                print(f"-------{value}")
                                  
                                  
    def outputBoundaryCondition(self):
        # Output the start of the 'boundary' section in blockMeshDict
        generate_boundary_start(self.output)
                                  
        # Output each boundary partitioning in blockMeshDict
        for key in self.boundaryDict:
            B_name = key
            B_type = self.boundaryDict[B_name]['type']
            generate_boundary_define_start(self.output, B_name, B_type)
            
            for i in range(1, self.boundaryDict[B_name]['faceNum']+1):
                face_point = self.boundaryDict[B_name][f'face_{i}']
                face_restricted_point = self.boundaryDict[B_name][f'faceRestricted_{i}']
                face_point = generate_coordIndex(self.coordinate_x, self.coordinate_y, self.coordinate_z, face_point)
                face_restricted_point = generate_coordIndex(self.coordinate_x, self.coordinate_y, self.coordinate_z, face_restricted_point)
                generate_face_index(self.numAreas_x, self.numAreas_y, self.numAreas_z, 
                                    self.output, face_point, self.if_expansion, face_restricted_point)
                
            generate_boundary_define_end(self.output)
        
        # Output the end of the 'boundary' section in blockMeshDict
        generate_boundary_end(self.output)
        print(f"Boundary partitioning has been output to --{self.output}-- folder")