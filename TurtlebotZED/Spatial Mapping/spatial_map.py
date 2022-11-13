import sys
import time
import pyzed.sl as sl
import viewer as gl

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
camera_parameters = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

pymesh = sl.Mesh()
image = sl.Mat()
pose = sl.Pose()

viewer = gl.GLViewer()
viewer.init(camera_parameters, pymesh)

tracking_params = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking()
if err != sl.ERROR_CODE.SUCCESS:
	exit(1)    

mapping_params = sl.SpatialMappingParameters()
err = zed.enable_spatial_mapping()
if err != sl.ERROR_CODE.SUCCESS:
    	exit(1)
    
tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

runtime_params = sl.RuntimeParameters()

last_call = time.time()

while viewer.is_available():
	if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
		# Retrieve left image
		zed.retrieve_image(image, sl.VIEW.LEFT)
		# Update pose data (used for projection of the mesh over the current image)
		tracking_state = zed.get_position(pose)
		
		# Extract whole mesh
		mapping_state = zed.get_spatial_mapping_state()
		
		duration = time.time() - last_call  
		# Ask for a mesh update if 500ms elapsed since last request
		if(duration > .5 and viewer.chunks_updated()):
			zed.request_spatial_map_async()
			last_call = time.time()
		
		if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
			zed.retrieve_spatial_map_async(pymesh)
			viewer.update_chunks()
		
		change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)
		
		# Configure spatial mapping parameters
		mapping_params.resolution_meter = mapping_params.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
		mapping_params.use_chunk_only = True 
		mapping_params.save_texture = True         # Set to True to apply texture over the created mesh
		mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH # Either MESH or FUSE_POINT_CLOUD
		
		zed.enable_spatial_mapping(mapping_params)
		pymesh.clear()
		viewer.clear_current_mesh()
		
		
		# Extract whole mesh
		zed.extract_whole_spatial_map(pymesh)
		filter_params = sl.MeshFilterParameters()
		filter_params.set(sl.MESH_FILTER.MEDIUM)
		print(mapping_params.save_texture)
		
		pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)
		
		# Filter the extracted mesh
		pymesh.filter(filter_params, True)
		
		pymesh.save('mesh.obj') 
		
		viewer.clear_current_mesh()
		last_call = time.time()
		

pymesh.clear()
zed.disable_spatial_mapping()
zed.disable_positional_tracking()
zed.close()
