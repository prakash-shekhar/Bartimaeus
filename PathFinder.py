# x_dim = 640
# y_dim = 480

# dimensions = (x_dim, y_dim)

# user_pos = (x_dim // 2, y_dim)


# def ccw(A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# def line_segment_intersection(p1, p2, p3, p4):
#     # returns true if line from p1 to p2 intersections with line from p3 to p4
#     # where p1 is user_pos, p2 is target_pos
#     return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


# def check_intersection(target_pos, user_pos, obstacle_x1, obstacle_y1, obstacle_x2, obstacle_y2, name):
#     obstacle_midpoint = ((obstacle_x1 + obstacle_x2) // 2, (obstacle_y1 + obstacle_y2) // 2)
#     obstacle_topleft = (obstacle_x1, obstacle_y1)
#     obstacle_topright = (obstacle_x2, obstacle_y1)
#     obstacle_bottomleft = (obstacle_x1, obstacle_y2)
#     obstacle_bottomright = (obstacle_x2, obstacle_y2)


#     L1_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_topleft)
#     R1_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_topright)
#     L2_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_bottomleft)
#     R2_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_bottomright)

#     if (L1_res):
#         return "Left"  # needed since there is an intersection

#     elif (R1_res):
#         return "Right"  # needed since there is an intersection

#     elif (L2_res):
#         return "Left"
#     elif (R2_res):
#         return "Right"



# def check_which_section(x_val):
#     print(0, x_dim // 3, (2 * (x_dim // 3)), x_dim, x_val)
#     if (0 <= x_val <= (x_dim // 3)):
#         return "left of screen"
#     elif ((x_dim // 3) < x_val <= (2 * (x_dim // 3))):
#         return "middle of screen"
#     elif ((2 * (x_dim // 3)) < x_val <= x_dim):
#         return "right of screen"


# def next_instruction(objects, target_name):
#     obstacle_list = []
#     target_midpoint = (-1, -1)
#     target_depth = -1
#     found_target = False

#     for obj in objects:
#         name, x1, y1, x2, y2, rel_depth = obj

#         if name == target_name:
#             # Find the center of the target box
#             target_midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
#             target_depth = rel_depth
#             # Ensure the target is on a walkable area
#             found_target = True


#             continue
#         else:
#             obstacle_list.append((name, x1, y1, x2, y2, rel_depth))

#             continue

#     if (not found_target):
#         #DMYolo will check if the previous outloud call based on the time delta value passes the second threshhold
#         return ("Turn around and look for the object", -1)

#     loc = check_which_section(target_midpoint[0])
#     print(loc)

#     obstacle_list = sorted(obstacle_list, key=lambda x: x[5], reverse=True)  # sorts it by closest objects

#     for obstacle in obstacle_list:
#         obstacle_depth = obstacle[5]

#         if (obstacle_depth < target_depth):
#             break

#         res = check_intersection(target_midpoint, user_pos, obstacle[1], obstacle[2], obstacle[3], obstacle[4],
#                                  obstacle[0])

#         #ONLY return an actual target_depth when the previous move is move forward so we can check the SECOND threshold for if we have reached the object now
#         if (res == "Left"):
#             if (loc == "middle of screen"):
#                 return (f"Turn Left, there is a {obstacle[0]} in front of you", -1)
#             elif (loc == "right of screen"):
#                 return ("Move Forward", target_depth)


#         elif (res == "Right"):
#             if (loc == "middle of screen"):
#                 return (f"Turn Right, there is a {obstacle[0]} in front of you", -1)
#             elif (loc == "left of screen"):
#                 return ("Move Forward", target_depth)

#         # no interseciton with this object
#     if (loc == "left of screen"):
#         return (f"Turn Left", -1)
#     elif (loc == "right of screen"):
#         return ("Turn Right", -1)

#     return ("Move Forward", target_depth)

# # also add when you said (based on depth >= some high threshold)"you have reached {target_name}"


##chatgpt

from utils import *

x_dim = 640
y_dim = 480

dimensions = (x_dim, y_dim)
user_pos = (x_dim // 2, y_dim)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_segment_intersection(p1, p2, p3, p4):
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def check_intersection(target_pos, user_pos, obstacle_x1, obstacle_y1, obstacle_x2, obstacle_y2, name):
    obstacle_midpoint = ((obstacle_x1 + obstacle_x2) // 2, (obstacle_y1 + obstacle_y2) // 2)
    obstacle_topleft = (obstacle_x1, obstacle_y1)
    obstacle_topright = (obstacle_x2, obstacle_y1)
    obstacle_bottomleft = (obstacle_x1, obstacle_y2)
    obstacle_bottomright = (obstacle_x2, obstacle_y2)

    L1_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_topleft)
    R1_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_topright)
    L2_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_bottomleft)
    R2_res = line_segment_intersection(user_pos, target_pos, obstacle_midpoint, obstacle_bottomright)

    if L1_res:
        return "Left"
    elif R1_res:
        return "Right"
    elif L2_res:
        return "Left"
    elif R2_res:
        return "Right"

def check_which_section(x_val):
    if 0 <= x_val <= (x_dim // 3):
        return "left of screen"
    elif (x_dim // 3) < x_val <= (2 * (x_dim // 3)):
        return "middle of screen"
    elif (2 * (x_dim // 3)) < x_val <= x_dim:
        return "right of screen"

def next_instruction(objects, target_name, depth_threshold=250, consecutive_frames=3):
    obstacle_list = []
    target_midpoint = (-1, -1)
    target_depth = -1
    found_target = False
    consecutive_target_frames = 0  # Tracks if we're consistently at the object

    for obj in objects:
        name, x1, y1, x2, y2, rel_depth = obj

        if name == target_name:
            target_midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            target_depth = rel_depth
            found_target = True
            continue
        else:
            obstacle_list.append((name, x1, y1, x2, y2, rel_depth))

    if not found_target:
        return "Turn around and look for the object", -1

    loc = check_which_section(target_midpoint[0])

    obstacle_list = sorted(obstacle_list, key=lambda x: x[5], reverse=True)

    for obstacle in obstacle_list:
        obstacle_depth = obstacle[5]

        if obstacle_depth < target_depth:
            break

        res = check_intersection(target_midpoint, user_pos, obstacle[1], obstacle[2], obstacle[3], obstacle[4], obstacle[0])

        if res == "Left":
            if loc == "middle of screen":
                speak_mac(f"Turn Left, there is a {obstacle[0]} in front of you")
                return f"Turn Left, there is a {obstacle[0]} in front of you", -1
            elif loc == "right of screen":
                speak_mac("Move Forward")
                return "Move Forward", target_depth
        elif res == "Right":
            if loc == "middle of screen":
                speak_mac(f"Turn Right, there is a {obstacle[0]} in front of you")
                return f"Turn Right, there is a {obstacle[0]} in front of you", -1
            elif loc == "left of screen":
                speak_mac("Move Forward")
                return "Move Forward", target_depth

    # Calculate if we're at the object by comparing target depth and frames at proximity
    if abs(target_depth - depth_threshold) <= 5:
        consecutive_target_frames += 1
        if consecutive_target_frames >= consecutive_frames:
            return f"You have reached {target_name}", target_depth
    else:
            consecutive_target_frames = 0

    if loc == "left of screen":
        return "Turn Left", -1
    elif loc == "right of screen":
        return "Turn Right", -1
    return ("Move Forward", target_depth)