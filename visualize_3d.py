import numpy as np
import cv2
import open3d as o3d

# Camera intrinsics for 1920x1080 resolution
original_fx, original_fy = 1000, 1000  # Focal lengths (approximate, adjust based on calibration if available)
original_cx, original_cy = 960, 540    # Principal point (center of the image)

fx, fy = original_fx / 4, original_fy / 4
cx, cy = original_cx / 4, original_cy / 4

# Load the depth map (replace this path with your actual depth map image)
original_depth_map = cv2.imread("depth_photos/00.png", cv2.IMREAD_UNCHANGED).astype(np.float32) 

# Resize the depth map to half its original dimensions
depth_map = cv2.resize(original_depth_map, (960, 540), interpolation=cv2.INTER_LINEAR)
print("done reading...")

# Generate the 3D point cloud
point_cloud = []
height, width = depth_map.shape
for v in range(height):
    for u in range(width):
        Z = depth_map[v, u]
        if Z == 0:  # Ignore points with no depth information
            continue

        # Calculate real-world coordinates X, Y, Z
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        point_cloud.append([X, Y, Z])

# Convert the point cloud to a NumPy array
point_cloud = np.array(point_cloud)
print("done processing...")

# Visualize the 3D point cloud with Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

print("visualizing...")
o3d.visualization.draw_geometries([pcd])