import os
import json
import numpy as np
import torch

def keep_path_last_n_names(path,n):
    """
    Keep only the last two levels of the given path.
    
    :param path: Original path
    :return: Path with only the last two levels
    """
    # Normalize the path to remove any redundant separators or up-level references
    normalized_path = os.path.normpath(path)
    
    # Split the path into parts
    path_parts = normalized_path.split(os.sep)
    
    # Keep only the last two levels
    last_two_levels = os.sep.join(path_parts[-n:])
    last_two_levels=last_two_levels.replace("/","_")
    last_two_levels=last_two_levels.replace("\\","_")
    return last_two_levels



def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.uint8:
            data=data[8:]
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data


def read_json_file(filepath):
    # if not os.path.exists(filepath):
    #     raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

from PIL import Image
def save_segmentation_as_png(vortexsegmentationLabel, filename, upSample=1.0):

    """
    Saves a 2D binary segmentation as a PNG file.

    Parameters:
        vortexsegmentationLabel (numpy.ndarray): The segmentation array of shape (Ydim, Xdim, 2).
        filename (str): The filename to save the PNG image.
        upSample (float): Upsampling factor to resize the image. Default is 1.0 (no scaling).
    """
    # Create the directory if it does not exist
    folder = os.path.dirname(filename)  # Extract the folder path from the filename
    if folder and not os.path.exists(folder):  # Ensure folder is non-empty and doesn't exist
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    
    # Convert the segmentation to a binary mask
    if len(vortexsegmentationLabel.shape)==3:
        binary_mask = np.where(vortexsegmentationLabel[..., 1] > 0.5, 255, 0).astype(np.uint8)
        
    binary_mask = np.where(vortexsegmentationLabel > 0.5, 255, 0).astype(np.uint8)
    
    # Create an image from the binary mask
    image = Image.fromarray(binary_mask, mode='L')  # 'L' mode for (8-bit pixels, black and white)
    
    # Apply upsampling if needed
    if upSample != 1.0:
        new_size = (int(image.width * upSample), int(image.height * upSample))
        image = image.resize(new_size, Image.NEAREST)  # Use NEAREST for upsampling binary images
    
    # Save the image
    image.save(filename)
def loadOneFlowEntrySteadySegmentation(binPath,Xdim,Ydim,Rawfeatures):

    raw_Binary = read_binary_file(binPath)
    fieldData = raw_Binary.reshape( Ydim,Xdim, -1)
    fieldData=fieldData[:,:,0:2]
    segmentation_Binary_path = binPath.replace('.bin','_segmentation.bin' )
    vortexsegmentationLabel = read_binary_file(segmentation_Binary_path,dtype=np.uint8).reshape(Ydim,Xdim).astype(np.float32)
    return fieldData,vortexsegmentationLabel

def loadOneFlowEntryCulrIVDSteadySegmentation(binPath,Xdim,Ydim):
    raw_Binary = read_binary_file(binPath)
    fieldData = raw_Binary.reshape( Ydim,Xdim, 4)
    segmentation_Binary_path = binPath.replace('.bin','_segmentation.bin' )
    vortexsegmentationLabel = read_binary_file(segmentation_Binary_path,dtype=np.uint8).reshape(Ydim,Xdim).astype(np.float32)
    return fieldData,vortexsegmentationLabel


def loadVastisFlowEntrySteadySegmentation(metaPath,Xdim,Ydim,domainMinBoundary,dominMaxBoundary):
    #get meta information
    metaINFo=read_json_file(metaPath)
    bin_file = metaPath.replace('meta.json','.bin' )
    
    raw_Binary = read_binary_file(bin_file)
    fieldData = raw_Binary.reshape( Ydim,Xdim, 2)

    if 'rc_n_si' in metaINFo and 'txy' in metaINFo  and 'deform_theta_sx_sy' in metaINFo:
        rc,n,si=metaINFo['rc_n_si']["value0"],metaINFo['rc_n_si']["value1"],metaINFo['rc_n_si']["value2"]
        txy=np.array([metaINFo['txy']["value0"],metaINFo['txy']["value1"]]  )  
        theta,sx,sy=metaINFo['deform_theta_sx_sy']["value0"],metaINFo['deform_theta_sx_sy']["value1"],metaINFo['deform_theta_sx_sy']["value2"]
        deformMatA=np.array([
                        [sx*np.cos(theta), -sy*np.sin(theta)],
                        [sx*np.sin(theta), sy*np.cos(theta)]])
        InvA= np.linalg.inv(deformMatA)
        
        
        gridInterval=[ float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Xdim-1),
                    float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Ydim-1)    ]
        vortexsegmentationLabel = np.zeros((Ydim, Xdim),dtype=np.float32)
        if si==0.0:
            vortexsegmentationLabel[:, :] = np.array([0.0], dtype=np.float32)
        else:
            for y in range(Ydim):
                for x in range(Xdim):
                    # Extract position vector from fieldData
                    pos=np.array([domainMinBoundary[0]+x*gridInterval[0],domainMinBoundary[1]+y*gridInterval[1]])
                    transformed_pos=np.dot(InvA, (pos-txy))
                    dx=rc- np.linalg.norm(transformed_pos)  
                    #dx>0 ->rc >distance->inside vortex->label to one()[0,1]
                    vortexsegmentationLabel[y, x] = np.array([1.0],dtype=np.float32) if dx>0 else np.array([0.0],dtype=np.float32) 
                    
    else:
        #try read binary segmetnation file directly
        segmentation_Binary_path = metaPath.replace('meta.json','_segmentation.bin' )
        vortexsegmentationLabel = read_binary_file(segmentation_Binary_path,dtype=np.uint8).reshape(Ydim,Xdim).astype(np.float32)
        
    return fieldData,vortexsegmentationLabel


def loadOneFlowEntryRawDataSteady(binPath,Xdim,Ydim):
    raw_Binary = read_binary_file(binPath)
    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)
    n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    fieldData = raw_Binary.reshape( Ydim,Xdim, 2)
    #si=0 means this is saddle not vortex, logits 0 means non-votex class, logits 1 means vortex class
    if si==0.0:
        vortexLabelOneHot= np.array([1.0,0.0],dtype=np.float32)
    else:
        vortexLabelOneHot=np.array([0.0,1.0],dtype=np.float32)
    return fieldData,vortexLabelOneHot


def loadOneFlowEntryRawData(binPath,Xdim,Ydim,time_steps):
    raw_Binary = read_binary_file(binPath)
    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)

    abc_tInfo=[value for value in metaINFo['observer_abc'].values()]
    abcdot_t=[value for value in metaINFo['observer_abc_dot'].values()] 
    referenceLabel=abc_tInfo+abcdot_t
                            
    # n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    # tx,ty=metaINFo['txy']["value0"],metaINFo['txy']["value1"]
    # vortexLableData= np.array([tx,ty,n,rc],dtype=np.float32) 
    #check raw_Binary.size
    assert (raw_Binary.size==time_steps*Ydim*Xdim* 2)
    fieldData = raw_Binary.reshape( time_steps,Ydim,Xdim, 2)
    referenceLabel= np.array(referenceLabel,dtype=np.float32)
    return fieldData,referenceLabel



def pad_or_truncate_pathlines(pathlineClusters, L):
    K = len(pathlineClusters)  
    C = len(pathlineClusters[0][0]) if pathlineClusters and pathlineClusters[0] else 0  

    result = np.zeros((L, K , C), dtype=np.float32)
    for i, pathline in enumerate(pathlineClusters):
        pathline_length = min(len(pathline), L)  
        for step in range(pathline_length):
            result[step, i,:] = np.array(pathline[step])
    result[0,:,-1]=0.0
    return result


def getClassificationOfPatch(si):
    #classfication of this patch
    vortexsegmentationLabel = np.zeros((1),dtype=np.float32)
    if si ==1.0 or si == 2.0 :
        vortexsegmentationLabel[0] = 1.0        
    else:
        # vortexsegmentationLabel[0] = 0.0 
        vortexsegmentationLabel[0] =0.0
    return vortexsegmentationLabel


def getSegmentationofPathlines(
        pathlineClusters, si=None, use_embedded_seed_labels=False):
    """Extract seed labels, preserving legacy profile-tag compatibility."""
    Klines, _pointsPerline, featurePerPoint = pathlineClusters.shape
    if featurePerPoint <= 4:
        raise ValueError("Pathline feature 4 is required for the seed label.")
    embedded_labels = np.asarray(
        pathlineClusters[:, 0, 4], dtype=np.float32).copy()
    if use_embedded_seed_labels or si == 1.0 or si == 2.0:
        if not np.all(np.isfinite(embedded_labels)) or np.any(
                (embedded_labels < 0.0) | (embedded_labels > 1.0)):
            raise ValueError(
                "Pathline seed labels must be finite values in [0, 1].")
        vortexsegmentationLabel = embedded_labels
    else:
        vortexsegmentationLabel = np.zeros((Klines), dtype=np.float32)
    # Remove the embedded target from the model input for every sample type.
    pathlineClusters[:, 0, 4] = 0.0
    return vortexsegmentationLabel

#point features: px, py, time, ivd(curl), distance, v_0, v_1([nablaV...])
feature_mapping = {
    'ivd': 3,
    'curl': 3,
    'abscurl': 3,
    'distance': 4,
    'velocity': [5, 6],
    # 'nablaV': slice(7, None)
}
def mask_out_addtional_feature(input_pathlineClusters, mask_out_features):
    for feature in mask_out_features:
        if feature in feature_mapping:
            if isinstance(feature_mapping[feature], int):                
                input_pathlineClusters[:, :, feature_mapping[feature]] = 0
            elif isinstance(feature_mapping[feature], list):
                for idx in feature_mapping[feature]:
                    input_pathlineClusters[:, :, idx] = 0                        
    return input_pathlineClusters    
    
    
def SpatialDownSampling(in_pathline_src,in_labels, downsample_ratio, linesPerGroup=4):
    L_Full_length, K, C = in_pathline_src.shape
    total_groups = K // linesPerGroup
    keepGroups: int = max(1, int(downsample_ratio * total_groups))
    # Calculate the stride to evenly distribute the groups
    stride = max(1, total_groups // keepGroups)
    
    # Generate indices for the groups to keep
    group_indices = torch.arange(0, total_groups, step=stride)[:keepGroups]
    
    # Create a mask for the lines to keep
    mask = torch.zeros(K, dtype=torch.bool)
    for idx in group_indices:
        start = idx * linesPerGroup
        end = min(start + linesPerGroup, K)  # Ensure we don't go out of bounds
        mask[start:end] = True
    
    sampled_pathline = in_pathline_src[:, mask, :]
    labels=in_labels[mask]
    return sampled_pathline,labels
          

def _load_pathline_velocity_field(
    raw_binary_path,
    Xdim,
    Ydim,
    time_steps,
    velocity_channel_names=None,
    velocity_binary_layout=None,
):
    """Load either the legacy unsteady field or the generator's t0 field.

    Legacy pathline datasets stored ``[T,Y,X,2]``.  The current generator
    stores one visualization frame as ``[Y,X,4]`` with channels
    ``[u,v,curl,IVD]``.  The returned layout is always ``[T,Y,X,2]`` so the
    dataset can convert it to the established ``[C,Y,X,T]`` tensor layout.
    """
    raw_binary = read_binary_file(raw_binary_path)
    spatial_size = int(Xdim) * int(Ydim)
    legacy_size = int(time_steps) * spatial_size * 2
    normalized_layout = str(velocity_binary_layout or "").lower().replace(" ", "")
    metadata_declares_frame = "[y,x,channel]" in normalized_layout

    if not metadata_declares_frame and raw_binary.size == legacy_size:
        return raw_binary.reshape(int(time_steps), int(Ydim), int(Xdim), 2)

    if spatial_size <= 0 or raw_binary.size % spatial_size != 0:
        raise ValueError(
            f"Unsupported velocity payload in {raw_binary_path}: got "
            f"{raw_binary.size} float32 values; expected either "
            f"T*Y*X*2={legacy_size} or Y*X*C with Y*X={spatial_size}."
        )

    channel_count = raw_binary.size // spatial_size
    if channel_count < 2:
        raise ValueError(
            f"Velocity payload in {raw_binary_path} has only "
            f"{channel_count} channel(s); at least u and v are required."
        )

    frame = raw_binary.reshape(int(Ydim), int(Xdim), channel_count)
    u_index, v_index = 0, 1
    if velocity_channel_names:
        normalized_names = [str(name).lower() for name in velocity_channel_names]
        if "velocity_u" in normalized_names and "velocity_v" in normalized_names:
            u_index = normalized_names.index("velocity_u")
            v_index = normalized_names.index("velocity_v")
            if max(u_index, v_index) >= channel_count:
                raise ValueError(
                    f"Velocity channel metadata for {raw_binary_path} refers "
                    f"to channel {max(u_index, v_index)}, but the payload has "
                    f"only {channel_count} channels."
                )

    # The generator only persisted frame zero.  Keep an explicit time axis;
    # do not manufacture the missing T-1 frames.
    velocity_frame = np.ascontiguousarray(frame[..., [u_index, v_index]])
    return velocity_frame[None, ...]


def loadUnsteadyFlowPathlineSegmentation(
    metaPath,
    PathlineLength,
    PathlineCount,
    PathlineFeature,
    downSampleRatio,
    mask_out_feature,
    mode="train",
    rawBinaryPath=None,
    Xdim=None,
    Ydim=None,
    time_steps=None,
    velocity_channel_names=None,
    velocity_binary_layout=None,
    use_embedded_seed_labels=False,
):
    if "saddle" in metaPath or "zero" in metaPath:
        legacy_profile_type = 0.0
    else:
        legacy_profile_type = 1.0
    fieldData=np.zeros([1,1,1,1],dtype=np.float32)
    if mode == "test" and rawBinaryPath is not None:
        if Xdim is None or Ydim is None or time_steps is None:
            raise ValueError(
                "Xdim, Ydim and time_steps are required when loading a "
                "velocity field for pathline visualization."
            )
        fieldData = _load_pathline_velocity_field(
            rawBinaryPath,
            Xdim=Xdim,
            Ydim=Ydim,
            time_steps=time_steps,
            velocity_channel_names=velocity_channel_names,
            velocity_binary_layout=velocity_binary_layout,
        )
    
    pathlineBinarypath= metaPath.replace('meta.json', '_pathline.bin')
    pathlineClusters=read_binary_file(pathlineBinarypath)
    pathlineClusters=pathlineClusters.reshape(PathlineCount,PathlineLength,PathlineFeature)
    
    vortexsegmentationLabel = getSegmentationofPathlines(
        pathlineClusters,
        legacy_profile_type,
        use_embedded_seed_labels=use_embedded_seed_labels,
    )
    # Permute pathline clusters first and second axis
    pathlineClusters = np.transpose(pathlineClusters, (1, 0, 2))    
    pathlineClusters,vortexsegmentationLabel=SpatialDownSampling(pathlineClusters,vortexsegmentationLabel,downsample_ratio=downSampleRatio,linesPerGroup=4)
    #for drop information  experiment.
    if mask_out_feature is not None:
        pathlineClusters=mask_out_addtional_feature(pathlineClusters,mask_out_features=mask_out_feature)
    
    # down sample input
    # L_Full_length, K, C = pathlineClusters.shape
    # linesPerGroup=5
    # total_groups = K // linesPerGroup
    # group_indices = torch.arange(0, total_groups, step=2)[:keepGroups]
    # mask = torch.zeros(K, dtype=torch.bool)
    # for idx in group_indices:
    #     start = idx * linesPerGroup
    #     end = start + linesPerGroup
    #     mask[start:end] = True
    # sampled_pathline = in_pathline_src[:, :, mask, :]

    
    return (fieldData,pathlineClusters),vortexsegmentationLabel



















# Generate grid of positions
    # x_coords = np.linspace(domainMinBoundary[0], dominMaxBoundary[0], Xdim)
    # y_coords = np.linspace(domainMinBoundary[1], dominMaxBoundary[1], Ydim)
    # X, Y = np.meshgrid(x_coords, y_coords)

    # # Create the position vectors
    # pos = np.stack([X, Y], axis=-1)  # Shape: (Ydim, Xdim, 2)
    
    # # Transform positions
    # transformed_pos = np.dot(pos - txy, InvA.T)  # Shape: (Ydim, Xdim, 2)

    # # Compute distances from the vortex center
    # distances = np.linalg.norm(transformed_pos, axis=-1)  # Shape: (Ydim, Xdim)

    # # Determine labels based on distance
    # inside_vortex = (rc > distances)
    # vortexsegmentationLabel[:, :, 1] = inside_vortex.astype(np.float32)
    # vortexsegmentationLabel[:, :, 0] = (~inside_vortex).astype(np.float32)
    
# vortexsegmentationLabel2 = np.zeros((Ydim, Xdim,2),dtype=np.float32)       
# gridInterval=[ float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Xdim-1),
#               float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Ydim-1)    ]
# if si==0.0:
#     vortexsegmentationLabel[:, :, 0] = 1.0
# else:
#     for y in range(Ydim):
#         for x in range(Xdim):
#             # Extract position vector from fieldData
#             pos=np.array([domainMinBoundary[0]+x*gridInterval[0],domainMinBoundary[1]+y*gridInterval[1]])
#             transformed_pos=np.dot(InvA, (pos-txy))
#             dx=rc- np.linalg.norm(transformed_pos)  
#             #dx>0 ->rc >distance->inside vortex->label to one()[0,1]
#             vortexsegmentationLabel[y, x] = np.array([0.0,1.0],dtype=np.float32) if dx>0.0 else np.array([1.0,0.0],dtype=np.float32)
