from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from .metrics import get_fold
from .action_simulator import initialize_simulator
from .action_simulator import SimulationStatus
from .simulator import check_for_occlusions
from .simulation_cache import get_default_100k_cache
from .objects_util import featurized_objects_vector_to_raster
from .vis import observations_to_float_rgb
import torch
import torch.nn as nn
import cv2
import sys
sys.path.append("../")
sys.path.append(".")
from dynamics.tools import gen_single_task_action

import phyre.interface.scene.ttypes as scene_if
import os
import pandas as pd
import pickle as pk
from collections import deque
import time
import seaborn as sns
import math
from scipy.ndimage import binary_dilation

TRAIN_SIZE = 1600
DEV_SIZE = 400
TEST_SIZE = 500

TASK_FORMAT='TEMPLATE_ID:INSTANCE_ID'

MAX_ITERATIONS = 100
MAX_TEST_ACTIONS = 10000
USE_EUCLIDEAN_DISTANCE=True

def observe_collision_events(simulator, action, noise=False, need_images=False):
    _, is_valid = simulator._get_user_input(action)
    simulation = simulator.simulate_action(task_index=0, 
                                           action=action, 
                                           need_images=need_images, 
                                           stride=20,
                                           noise=noise, 
                                           need_featurized_objects=True, 
                                           need_collisions=True)
    
    x,y = simulation.featurized_objects.xs * 256, simulation.featurized_objects.ys * 256
    angles, diameters = simulation.featurized_objects.angles, simulation.featurized_objects.diameters
    diameters = np.tile(diameters, (angles.shape[0], 1)) * 256
    colors = simulation.featurized_objects.colors

    path = np.array(list(zip(x,y,angles,diameters))).transpose((2,0,1))
    path_dict = {}
    
    for ii in range(len(path)):
        if colors[ii] not in ["RED", "BLACK"]:
            path_dict[ii] = path[ii]
        elif colors[ii] == "RED":
            path_dict['PLACED'] = path[ii]
    collisions = simulation.collisions
    
    return path_dict, simulation.status, collisions, None if not need_images else simulation.images

def get_goal_objs_ids(simulator):
    obj1_id = simulator._tasks[0].bodyId1 - 4
    obj2_id = simulator._tasks[0].bodyId2 - 4
    return (obj1_id, obj2_id)

def action_to_user_input(simulator, action):
    user_input, valid = simulator._action_mapper.action_to_user_input(action)
    occlusion = check_for_occlusions(simulator._serialized[0],user_input)
    return user_input, valid and not occlusion

def scale_action(simulator, action):
    user_input, valid = action_to_user_input(simulator, action)
    if valid:
        return (user_input.balls[0].position.x, user_input.balls[0].position.y, user_input.balls[0].radius)
    else:
        return (-1,-1,-1)

def embed_tool(action, puzzle_pixels):
    image = puzzle_pixels.copy()
    x,y = action["position"]
    # invert y axis
    y = 256 - y
    radius = action["radius"]
    min_x, min_y, max_x, max_y = x - radius, y - radius, x + radius, y + radius
    image = cv2.circle(image, (x,y), radius, color=(0.95294118, 0.30980392, 0.2745098), thickness=-1)
    # image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
    return image, np.array([min_x, min_y, max_x, max_y])

def get_initial_state(simulator):
    image = observations_to_float_rgb(simulator._initial_scenes[0])
    objs_color = simulator._initial_featurized_objects[0].colors
    objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
    objs = simulator._initial_featurized_objects[0].features[:, objs_valid, :]
    num_objs = objs.shape[1]
    bboxes = np.zeros((num_objs, 4))
    
    for o_id in range(num_objs):
        mask = featurized_objects_vector_to_raster(objs[0][[o_id]])
        mask_im = observations_to_float_rgb(mask)
        mask_im[mask_im == 1] = 0
        mask_im = mask_im.sum(-1) > 0

        [h, w] = np.where(mask_im)

        assert len(h) > 0 and len(w) > 0
        x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
        bboxes[o_id] = [x1, y1, x2, y2]
    return image, bboxes

def get_global_min_dist(simulator):
    obj1_id, obj2_id = get_goal_objs_ids(simulator)
    action = [-1,-1,-1]
    path_dict, _, _, images = observe_collision_events(simulator, action, False, True)
    distance = get_min_distance_between_objects(simulator, obj1_id, obj2_id, path_dict, images)
    return distance

def intersect(A,B,C,D):
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def min_distance_point_to_segment(segment, p):
    p = np.array(p)
    r = segment[1] - segment[0]
    a = segment[0] - p

    min_t = np.clip(-a.dot(r) / (r.dot(r)), 0, 1)
    d = a + min_t * r

    return np.sqrt(d.dot(d))

def get_line_endpoints(middle_point, angle_normalized, length):
    angle_rad = angle_normalized * 2 * np.pi
    half_length = length / 2
    
    dx = np.cos(angle_rad) * half_length
    dy = np.sin(angle_rad) * half_length
    
    x, y = middle_point
    return np.array([
        [x - dx, y - dy],  # Start point
        [x + dx, y + dy]   # End point
    ])

def clear_path_between_goal_objects(image):
    green_mask = (image == 2)                   
    blue_purple_mask = (image == 3) | (image == 4)
    obstacle_mask = (image == 0)

    while not np.any(blue_purple_mask & green_mask):
        blue_purple_mask = binary_dilation(blue_purple_mask)
        blue_mask_with_obstacles = blue_purple_mask.copy()
        blue_mask_with_obstacles[obstacle_mask] = False

        green_mask = binary_dilation(green_mask)
        green_mask_with_obstacles = green_mask.copy()
        green_mask_with_obstacles[obstacle_mask] = False

        # If either region hits an obstacle, return False
        if not np.array_equal(blue_purple_mask, blue_mask_with_obstacles) or not np.array_equal(green_mask, green_mask_with_obstacles):
            return False
        # Stop if the blue and green regions touch
        if np.any(blue_purple_mask & green_mask):
            return True
    return True

def get_min_distance_between_objects(simulator, obj1_id, obj2_id, path_dict, images):

    obj1_type = get_obj_type(simulator, obj1_id)
    obj2_type = get_obj_type(simulator, obj2_id)

    min_distance = np.inf
    window_len = 9
    window = deque(maxlen=window_len)

    if obj1_type in ["ball","jar"] and obj2_type in ["ball","jar"]:
        radius = []
        if obj1_type == 'ball':
            radius.append(path_dict[obj1_id][0,3]/2)
        else:
            radius.append(path_dict[obj1_id][0,3]/4)
        if obj2_type == 'ball':
            radius.append(path_dict[obj2_id][0,3]/2)
        else:
            radius.append(path_dict[obj2_id][0,3]/4)
        obj1_path = path_dict[obj1_id][:,:2][::-1] # path in reverse to avoid bouncing
        obj2_path = path_dict[obj2_id][:,:2][::-1]
        radius = path_dict[obj1_id][0,3]/4, path_dict[obj2_id][0,3]/4
        distances = np.linalg.norm(obj1_path - obj2_path, axis=-1) - radius[0] - radius[1]
        distances[distances < 0] = 0.0
        timestep = np.argmin(distances)

        while timestep < (len(distances) - 2) and distances[timestep+1] == distances[timestep]:
            timestep += 1
        min_index = max(0, timestep - window_len) # we are going backwards
        if min_index == timestep:
            timestep += 1
        obstacle_padding = 0 # if clear_path_between_goal_objects(images[timestep]) else 50
        time_padding = 0 if (timestep - min_index) == window_len else 3 * (window_len -(timestep-min_index))
        ma_min = obstacle_padding + time_padding + (distances[timestep] if min_index == timestep else sum(distances[min_index:timestep])/(timestep-min_index))
        

    elif obj1_type in ["bar", "stick"] and obj2_type in ["bar", "stick"]:
        # distance two lines --> shortest distance between endpoint and line
        obj1_endpoints = [get_line_endpoints((x,y), angle, length) for x,y,angle,length in path_dict[obj1_id][::-1]]
        obj2_endpoints = [get_line_endpoints((x,y), angle, length) for x,y,angle,length in path_dict[obj2_id][::-1]]
        
        for t in range(len(obj1_endpoints)):
            obj1_start, obj1_end = obj1_endpoints[t]
            obj2_start, obj2_end = obj2_endpoints[t]

            # check if lines intersect
            lines_intersect = intersect(obj1_start, obj1_end, obj2_start, obj2_end)
            
            if lines_intersect:
                min_distance = 0
                ma_min = 0
                break

            # Compute minimum distance between segments by checking all point-to-segment combinations
            d1 = min_distance_point_to_segment([obj2_start, obj2_end], obj1_start)
            d2 = min_distance_point_to_segment([obj2_start, obj2_end], obj1_end)
            d3 = min_distance_point_to_segment([obj1_start, obj1_end], obj2_start)
            d4 = min_distance_point_to_segment([obj1_start, obj1_end], obj2_end)

            timestep_min = min(d1, d2, d3, d4)
            window.append(timestep_min)

            if timestep_min <= min_distance:
                min_distance = timestep_min
                timestep = t
                
                while timestep < (len(window) - 2) and window[timestep+1] == window[timestep]:
                    timestep += 1
                min_index = max(0, timestep - window_len) # we are going backwards
                if min_index == timestep:
                    timestep += 1

                time_padding = 0 if (timestep - min_index) == window_len else 3 * (window_len -(timestep-min_index))
                obstacle_padding = 0 # if clear_path_between_goal_objects(images[timestep]) else 50
                ma_min = obstacle_padding + time_padding + (window[timestep] if min_index == timestep else sum(window)/len(window))
                

    elif obj1_type in ["bar", "stick"] and obj2_type == "ball" or obj2_type in ["bar", "stick"] and obj1_type == "ball":
        if obj1_type == "ball":
            radius = path_dict[obj1_id][0,3]/2
            ball_endpoints = path_dict[obj1_id][:,:2][::-1]
            segment_endpoints = [get_line_endpoints((x,y), angle, length) for x,y,angle,length in path_dict[obj2_id][::-1]]
        else:
            radius = path_dict[obj2_id][0,3]/2
            ball_endpoints = path_dict[obj2_id][:,:2][::-1]
            segment_endpoints = [get_line_endpoints((x,y), angle, length) for x,y,angle,length in path_dict[obj1_id][::-1]]

        for t in range(len(ball_endpoints)):
            distance = min_distance_point_to_segment(segment_endpoints[t], ball_endpoints[t]) - radius
            window.append(distance)
            
            if distance <= min_distance:
                min_distance = distance
                timestep = t
                while timestep < (len(window) - 2) and window[timestep+1] == window[timestep]:
                    timestep += 1
                min_index = max(0, timestep - window_len) # we are going backwards
                if min_index == timestep:
                    timestep += 1

                time_padding = 0 if (timestep - min_index) == window_len else 3 * (window_len -(timestep-min_index))
                obstacle_padding = 0 # if clear_path_between_goal_objects(images[timestep]) else 50
                ma_min = obstacle_padding + time_padding + (window[timestep] if min_index == timestep else sum(window)/len(window))
                
    else:
        raise Exception("Shape combination not found")
    print(obj1_type, obj2_type)
    return max(0.0, ma_min)

def min_dist_from_image_mask(image):
    # 0 white, 1 red, 2 green, 3 blue, 4 purple, 5 gray, 6 black

    free_space_mask = (image == 0)
    green_mask = (image == 2)                   
    blue_purple_mask = (image == 3) | (image == 4)

    # Get coordinates 
    green_coords = np.argwhere(green_mask)
    blue_coords = np.argwhere(blue_purple_mask)

    rows, cols = free_space_mask.shape
    dist = np.full((rows, cols), float('inf'))

    queue = deque()
    for x, y in green_coords:
        dist[x, y] = 0
        queue.append((x, y))

    combined_mask = free_space_mask | blue_purple_mask

    # Perform BFS to propagate distances
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        current_distance = dist[x, y]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and combined_mask[nx, ny]:
                if dist[nx, ny] > current_distance + 1:
                    dist[nx, ny] = current_distance + 1
                    queue.append((nx, ny))

    # Find the shortest distance
    try:
        shortest_distance = min(dist[x, y] for x, y in blue_coords if dist[x, y] < float('inf'))
    except:
        # path is blocked
        return np.inf
    return shortest_distance

def get_min_dist(simulator, action=[-1,-1,-1], noise=False):
    path_dict, _, _, images = observe_collision_events(simulator, action, noise, True)
    distances = []
    min_dist = -1
    images = images[::-1]
    RESIZE = 64 if noise else 256

    if USE_EUCLIDEAN_DISTANCE:
        if action[0] == -1:
            dist = get_global_min_dist(simulator)
        else:
            obj1_id, obj2_id = get_goal_objs_ids(simulator)
            dist = get_min_distance_between_objects(simulator, obj1_id, obj2_id, path_dict, images)
        return dist
    
    for ii, image in enumerate(images[::2]):
        downsampled_image = cv2.resize(image, (RESIZE, RESIZE), interpolation=cv2.INTER_NEAREST)
        dist = min_dist_from_image_mask(downsampled_image)
        if dist == np.inf:
            distances.append(256*4)
        else:
            dist *= 256/RESIZE
            if min_dist < 0 or dist < distances[min_dist] or dist==distances[min_dist] and ii-min_dist == 1:
                min_dist = ii
            distances.append(dist)
    min_dist = distances[min_dist]

    print('Min dist for ', action, ':', min_dist)
    return min_dist

def get_obj_bbox(simulator, obj_id, timestep=0):
    inv_act = (-1,-1,-1) if 'two' not in simulator.tier else (-1,-1,-1,-1,-1,-1)
    _, _, bboxes, _ = get_task_gt_data(simulator, inv_act)
    for step in range(len(bboxes)):
        for obj in range(len(bboxes[step])):
            bboxes[step][obj][1:] = bboxes[step][obj][1:] * 256 / 128 
            # flip y axis
            bboxes[step][obj] = (obj, bboxes[step][obj][1], 256 - bboxes[step][obj][4], bboxes[step][obj][3], 256 - bboxes[step][obj][2])
    return bboxes[timestep][obj_id][1:]

def get_task_gt_data(simulator, action):
    full_images, status, bboxes, masks = gen_single_task_action(simulator, action, stride=20)
    return full_images, status, bboxes, masks

def get_puzzle_bboxes(simulator, action):
    _, status, bboxes, _ = get_task_gt_data(simulator, action)
    for step in range(len(bboxes)):
        for obj in range(len(bboxes[step])):
            bboxes[step][obj][1:] = bboxes[step][obj][1:] * 256 / 128
    
    return bboxes, status

def get_obj_type(simulator, obj_id):
    shape_one_hot = simulator._initial_featurized_objects[0].shapes_one_hot[obj_id]
    if shape_one_hot[0]:
        return "ball"
    elif shape_one_hot[1]:
        return "bar"
    elif shape_one_hot[2]:
        return "jar"
    elif shape_one_hot[3]:
        return "stick"
    else:
        raise Exception("Shape type not found.")


def get_simulator(setup, template_id, instance_id):
    simulator = initialize_simulator([str.zfill(template_id, 5) + ':' + str.zfill(instance_id, 3)], '_'.join(setup.split('_')[:-2]))
    return simulator

def get_tasks_ids_from_set(setup, fold, set):
    train_tasks, dev_tasks, test_tasks = get_fold(setup, fold)
    task_set = []
    if set == 'train':
        for task in train_tasks:
            task_set.append(task.split(":"))
    elif set == 'dev':
        for task in dev_tasks:
            task_set.append(task.split(":"))
    elif set == 'test':
        for task in test_tasks:
            task_set.append(task.split(":"))
    else:
        raise ValueError('Invalid set')
    return task_set

def execute_action(simulator, action, plot=False, save_gif=False, noise=False):
    task_index = 0 # if we are only running the simulator for 1 task
    simulation = simulator.simulate_action(task_index, action, need_images=plot, stride=10, noise=noise, need_featurized_objects=True, need_collisions=True)

    solved = True if simulation.status == SimulationStatus.SOLVED else False
    if plot:
        observations = []
        for img in simulation.images:
            observations.append(observations_to_float_rgb(img))
        observations = np.array(observations)
        fig, ax = plt.subplots()
        def update(idx):
            ax.imshow(observations[idx][::-1], origin='lower')
        ani = FuncAnimation(fig, update, frames=len(observations), interval=100)
        if save_gif:
            ani.save('animation.gif', writer='pillow')
        plt.show()
    return solved
    

def plot_task(simulator):
    task_index = 0 # if we are only running the simulator for 1 task
    initial_scene = simulator.initial_scenes[task_index]
    plt.imshow(observations_to_float_rgb(initial_scene)[::-1], origin='lower')
    plt.title(f'Task {simulator.task_ids[task_index]}')
    plt.show()

def plot_inference_result(simulator, action, bboxes, objects, ground_truth=False):
    pixels,_ = get_initial_state(simulator)
    user_input, valid = action_to_user_input(simulator, action)
    position = user_input.balls[0].position.x, user_input.balls[0].position.y
    radius = user_input.balls[0].radius
    pixels,_ = embed_tool({"position": position, "radius": radius}, pixels)
    if len(action) > 3:
        position = user_input.balls[1].position.x, user_input.balls[1].position.y
        radius = user_input.balls[1].radius
        pixels,_ = embed_tool({"position": position, "radius": radius}, pixels)
    
    # define cmap of colors as long as the number of objects
    # cmap = plt.cm.get_cmap('tab20', len(objects))

    _, ax = plt.subplots()
    plt.imshow(pixels)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

   
    for jj, step in enumerate(bboxes):
        for ii, bbox in enumerate(step):
            min_x, min_y, max_x, max_y = bbox
            if jj < 0:
                continue
            else:
                color = 'red' if isinstance(objects[ii],str) else 'blue'
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                ax.add_patch(plt.Circle(center, 2, fill=True, edgecolor=color, facecolor=color))
            if ii == len(objects)-1:
                break

    # draw lines connecting points
    for jj, step in enumerate(bboxes):
        for ii, bbox in enumerate(step):
            if jj < 1:
                continue
            min_x, min_y, max_x, max_y = bbox
            center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            prev_bbox = bboxes[jj-1][ii]
            prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
            color = 'red' if isinstance(objects[ii],str) else 'blue'
            ax.plot([prev_center[0], center[0]], [prev_center[1], center[1]], color=color, linewidth=1)
            if ii == len(objects)-1:
                break
    
    if ground_truth:
        # plot ground truth bounding boxes
        puzzle_bboxes,_ = get_puzzle_bboxes(simulator, action)
        for ii, obj in enumerate(objects[:puzzle_bboxes.shape[1]]):
            for step in range(0, len(puzzle_bboxes)):
                min_x, min_y, max_x, max_y = puzzle_bboxes[step][ii][1:]
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                ax.add_patch(plt.Circle(center, 2, fill=True, edgecolor='black', facecolor='black'))
        # draw lines connecting points
        for ii, obj in enumerate(objects[:puzzle_bboxes.shape[1]]):
            bbox = puzzle_bboxes[ii]
            for step in range(2, len(puzzle_bboxes)):
                min_x, min_y, max_x, max_y = puzzle_bboxes[step][ii][1:]
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                prev_bbox = puzzle_bboxes[step-1][ii][1:]
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                ax.plot([prev_center[0], center[0]], [prev_center[1], center[1]], color='black', linewidth=1)
        # ax.legend(['Ground truth'], loc='upper right', fontsize=16)
    filename = f'inference_result.svg'
    # plt.savefig(filename, dpi=300, bbox_inches='tight', format='svg')
    # plt.savefig(filename.replace('svg', 'png'), dpi=300, bbox_inches='tight', format='png')
    plt.show()

def get_velocity_vector_from_bboxes_phyre(simulator, args, bboxes, objects):
    velocity_vectors = []
    taken_actions = simulator.get_runs()
    # bboxes shape is (num_actions, num_steps, num_objects, 4)
    for jj, action in enumerate(bboxes):
        scaled_action = scale_action(simulator, args[jj])
        if scaled_action in taken_actions:
            velocity_vectors.append(taken_actions[scaled_action]['v_vec'])
            continue
        objs = objects[jj]
        # get object trajectories
        obj_traj = {}
        for ii, obj in enumerate(objs):
            bbox_seq = action[:,ii]
            center_seq = torch.tensor([((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2) for bbox in bbox_seq])
            obj_traj[obj] = center_seq

        v_vec = get_velocity_vector_from_trajectories_phyre(obj_traj, len(args[jj]))

        velocity_vectors.append(v_vec)
            
    return velocity_vectors

def get_velocity_vector_from_trajectories_phyre(obj_traj, action_size):
    STEPS_AHEAD = 10
    MAX_STEPS = 15
    v_vec = {}
    objs = list(obj_traj.keys())

    # np array to torch tensor
    for k, v in obj_traj.items():
        if not isinstance(v, torch.Tensor):
            obj_traj[k] = torch.tensor(v[:,:2], dtype=torch.float32)

    # check if tool was placed
    if 'PLACED' not in obj_traj:
        for ii in range(len(objs)):
            initial_center = obj_traj[objs[ii]][0][:2]
            final_center = obj_traj[objs[ii]][STEPS_AHEAD-1][:2]
            # if the movement is too small, ignore it (< 5 pixels)
            if torch.linalg.norm(final_center - initial_center) < 2:
                v_vec[objs[ii]] = torch.tensor([torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)])
                continue
            v_vec[objs[ii]] = (final_center[0] - initial_center[0], final_center[1] - initial_center[1])
        v_vec['PLACED'] = torch.tensor([torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)])
        return v_vec

    # get initial and final step if there is collision
    # otherwise, get first and STEPS_AHEAD steps ahead

    # get trajectories for each object
    tool_trajs = []
    if 'PLACED' in obj_traj:
        tool_trajs.append(obj_traj['PLACED'][:,:2])
    dynamic_objects_traj = {obj: obj_traj[obj][:,:2] for obj in objs if not isinstance(obj,str)}
    dynamic_objects = list(dynamic_objects_traj.keys())
    dynamic_objects_traj = [dynamic_objects_traj[obj] for obj in dynamic_objects]
    dynamic_objects_traj = torch.stack(dynamic_objects_traj, dim=0)

    # dynamic_objects_traj shape is (num_objects, num_steps, 2)
    # tool_traj shape is (num_steps, 2)
    # compute distances between tool and objects at each step (num_objects, num_steps)
    closest_point_per_obj = torch.zeros((dynamic_objects_traj.shape[0], 1), dtype=torch.int64)
    closest_point_per_obj_dist = torch.ones((dynamic_objects_traj.shape[0], 1), dtype=torch.float32) * 10000
    for tool_traj in tool_trajs:
        dist = torch.linalg.norm(tool_traj - dynamic_objects_traj, dim=2)
        closest_point_per_obj_dist_aux = torch.min(dist, dim=1).values
        closest_point_per_obj_aux = torch.argmin(dist, dim=1)
        for i in range(closest_point_per_obj.shape[0]):
            if closest_point_per_obj_dist_aux[i] < closest_point_per_obj_dist[i]:
                closest_point_per_obj_dist[i] = closest_point_per_obj_dist_aux[i]
                closest_point_per_obj[i] = closest_point_per_obj_aux[i]

    # get the point where any dynamic object collides with the tool (distance < 50 pixels)
    # then take the smallest point index (collision that happens first)

    colliding_points = torch.where(closest_point_per_obj_dist < 30)[0]
    if len(colliding_points) > 0:
        closest_point = torch.min(closest_point_per_obj[colliding_points])
        initial_step = min(closest_point, min(MAX_STEPS-1, dynamic_objects_traj.shape[1]))
        final_step = min(closest_point + STEPS_AHEAD, min(MAX_STEPS, dynamic_objects_traj.shape[1]))
    else:
        initial_step = 0
        final_step = STEPS_AHEAD

    if final_step - initial_step < 2:
        initial_step = 0
        final_step = STEPS_AHEAD

    for ii in range(len(dynamic_objects) + len(tool_trajs)):
        if ii == len(dynamic_objects):
            initial_center = tool_trajs[0][initial_step]
            final_center = tool_trajs[0][final_step - 1]
            o = 'PLACED'
        else:
            initial_center = dynamic_objects_traj[ii, initial_step]
            final_center = dynamic_objects_traj[ii, final_step-1]
            o = dynamic_objects[ii]

        # if the movement is too small, ignore it (< 5 pixels)
        if torch.linalg.norm(final_center - initial_center) < 2:
            v_vec[o] = torch.tensor([torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)])
            continue
            
        v_vec[o] = (final_center[0] - initial_center[0], final_center[1] - initial_center[1])

    return v_vec

def get_dynamic_objects(simulator):
    dynamic_objects = []
    bodies = simulator._tasks[0].scene.bodies
    static = 0
    for ii, body in enumerate(bodies):
        if body.bodyType == 2:
            dynamic_objects.append(ii - static)
        else:
            static += 1
    return dynamic_objects

def load_runs_data(setup, set, name, fold, run):
    parent_dir = '/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[:-3])
    path = os.path.join(parent_dir,"evals","phyre",'_'.join(setup.split('_')[:-1]),run)
    print(os.path.join(path, f"{set}_{name}.csv"))
    try:
        data = pd.read_csv(os.path.join(path, f"{set}_{name}.csv"))
    except Exception as e:
        return []
    # get iterations for the specified fold
    puzzle_trials = list(data[data.iloc[:,0] == fold]['iterations'])
    if len(puzzle_trials) == 0:
        puzzle_trials = list(data[data.iloc[:,0]== data.iloc[:,0][0]]['iterations'])
    # return get_runs_data(set, name)
    print(puzzle_trials)
    # puzzle_trials = list(np.array(puzzle_trials) + 9)
    print(puzzle_trials)
    return puzzle_trials

def compute_auccess(runs, max_iterations=MAX_ITERATIONS):
    k = runs[0]
    runs = list(np.cumsum(runs[1]))
    auccess = []
    numerator, denominator = 0.0, 0.0
    last_k = -1
    for i in range(1, max_iterations + 1):
        weight = np.log(i + 1) - np.log(i)
        denominator += weight
        if i not in k and last_k < 0:
            continue
        elif i in k:
            last_k += 1
        numerator += weight * runs[last_k]
        auccess.append(numerator / denominator)
    
    if len(auccess) == 0:
        return[0]
    return auccess

def get_puzzle_run_auccess(setup, set, name, fold, run):
    runs = np.array(load_runs_data(setup, set, name, fold, run))
    if len(runs) == 0:
        return -1
    # set all values > MAX_ITERATIONS to MAX_ITERATIONS + 1
    runs[runs > MAX_ITERATIONS] = MAX_ITERATIONS + 1
    runs = np.unique(runs, return_counts=True)
    total_runs = np.sum(runs[1])
    # divide counts by total to get probability
    runs = np.array([runs[0], runs[1] / total_runs])
    auccess = compute_auccess(runs,100)
    return auccess[-1]

def load_runs_humans_data(setup,set,name):
    path = os.path.join(".","evals","phyre","ball_cross","humans")
    data = pd.read_csv(os.path.join(path, f"all_trials.csv"))
    # filter by set and name
    taskid = set + ':' + name
    puzzle_trials = data[data["TaskID"] == taskid]
    if len(puzzle_trials) == 0:
        return []
    # group by user
    puzzle_trials = puzzle_trials.groupby("ID").agg({'AttemptNum':'max', 'SuccessTrial':'max'})[['AttemptNum','SuccessTrial']].reset_index()
    # get iterations
    failure = np.array(list(puzzle_trials['SuccessTrial'] == False))
    puzzle_trials = np.array(list(puzzle_trials['AttemptNum']))
    puzzle_trials[puzzle_trials > 11] = 11
    puzzle_trials[failure] = 11
    return puzzle_trials

def get_puzzle_run_human_auccess(setup, set, name):
    runs = np.array(load_runs_humans_data(setup, set, name))
    if len(runs) == 0:
        return -1
    # set all values > MAX_ITERATIONS to MAX_ITERATIONS + 1
    runs[runs > 10] = 10 + 1
    runs = np.unique(runs, return_counts=True)
    total_runs = np.sum(runs[1])
    # divide counts by total to get probability
    runs = np.array([runs[0], runs[1] / total_runs])
    auccess = compute_auccess(runs, 10)
    # idx = min(len(auccess)-1, 10)
    return auccess[-1]


def visualize_puzzle_runs_phyre(setup, set, name, compare_ssup=True):
    eval_runs = ['rbf_uniform']
    titles = ['RBF-uniform']
    color = (139/255,187/255,141/255)
    plt.figure(figsize=(6, 4))
    for run in eval_runs:
        runs = np.array(load_runs_data(setup, set, name, run))
        if len(runs) == 0:
            return -1
        # set all values > MAX_ITERATIONS to MAX_ITERATIONS + 1
        runs[runs > MAX_ITERATIONS] = MAX_ITERATIONS + 1
        runs = np.unique(runs, return_counts=True)
        total_runs = np.sum(runs[1])
        # divide counts by total to get probability
        runs = np.array([runs[0], runs[1] / total_runs])
        auccess = compute_auccess(runs)
        print('auccess', auccess[-1])
        x = list(runs[0])
        x.insert(0, 0)
        # compute cumsum of attempts 
        runs = list(np.cumsum(runs[1]))
        runs.insert(0, 0)
        if MAX_ITERATIONS not in x:
            x.append(MAX_ITERATIONS)
            runs.append(1.0)
        runs.insert(0, 0)
        plt.step(x, runs[:-1], label=titles[eval_runs.index(run)], color=color, linewidth=5)

    plt.xlabel("Attempts")
    plt.ylim(-0.03, 1.03)
    plt.xlim(-1, MAX_ITERATIONS+1)
    plt.grid()
    plt.title(f"Cumulative Solution Rate: {name}")
    # set font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    parent_dir = '/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[:-3])
    path = os.path.join(parent_dir,"evals","phyre",'_'.join(setup.split('_')[:-1]),run, "figures")
    plt.savefig(os.path.join(path, f"{set}_{name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(path, f"{set}_{name}.svg"), dpi=300, bbox_inches='tight', format='svg')

    plt.close()

def generate_kernel_cache(task_id, simulator, inference, PHYRE_SET):
    # get task cache if it exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, 'cache', 'kernel', task_id + '.npy')
    if os.path.exists(filename):
        print("Found cache:", filename)
        kernel = np.load(filename)
        return torch.tensor(kernel)
    else:
        print("Cache not found:", filename)
        actions = phyre_gen_candidates(task_id, PHYRE_SET)
        vvec_cache = generate_vvec_cache(task_id, simulator, inference)

        v = [vvec_cache[tuple(x.item() for x in a)] for a in actions]
        keys = list(v[0].keys())
        if not all(set(d.keys()) == set(keys) for d in v):
            raise ValueError("All dictionaries must have the same keys")
        tensors_1 = []
        for key in keys:
            vec_tensor = torch.stack([torch.Tensor(d[key]) for d in v])
            tensors_1.append(vec_tensor)
        
        tensors_1 = torch.stack(tensors_1)
        tensors_1 = torch.permute(tensors_1, (1, 0, 2))
        tensors_2 = tensors_1.clone()
        # Expand dimensions for broadcasting
        tensors_expanded_1 = tensors_1.unsqueeze(1)
        tensors_expanded_2 = tensors_2.unsqueeze(0)
        vector_sim = vector_similarity(tensors_expanded_1, tensors_expanded_2)
        dist = torch.mean(vector_sim, dim=-1)
        kernel = dist * torch.exp(dist-1)
        np.save(filename, kernel.detach().cpu().numpy())
        return kernel


def vector_similarity(u, v):
    SCALE_FACTOR = 100
    eps = 1e-6

    cos = nn.CosineSimilarity(dim=-1, eps=eps)
    cosine_sim = cos(u, v)

    u_norm = torch.linalg.norm(u, dim=-1)
    v_norm = torch.linalg.norm(v, dim=-1)

    zero_mask_u = u_norm <= eps
    zero_mask_v = v_norm <= eps

    both_zero_mask = zero_mask_u & zero_mask_v
    at_least_one_zero_mask = zero_mask_u | zero_mask_v

    magnitude_factor = 1 / (1 + torch.abs(u_norm - v_norm) / SCALE_FACTOR)

    similarities = cosine_sim * magnitude_factor
    similarities[both_zero_mask] = 1.0
    similarities[at_least_one_zero_mask & ~both_zero_mask] = 0.0

    # clamp negative values to 0
    return torch.clamp(similarities, min=0.0)

def generate_actions_cache(task_id, PHYRE_SET):
    cache = {}
    actions = phyre_gen_candidates(task_id, PHYRE_SET)
    for ii,a in enumerate(actions):
        cache[tuple(x.item() for x in a)] = ii
    return cache

def get_solution(task_id, PHYRE_SET):
    if PHYRE_SET != "2B":
        cache = get_default_100k_cache('ball')
    else:
        cache = get_default_100k_cache('two_balls')
    actions = cache.action_array[:MAX_TEST_ACTIONS].astype(np.double)
    statuses = cache.load_simulation_states(task_id)[:MAX_TEST_ACTIONS]
    solutions = actions[statuses==1]
    return solutions[0]

    
def generate_vvec_cache(task_id, simulator, inference, PHYRE_SET):
    # get task cache if it exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, 'cache', 'vvec',  task_id + '.pkl')
    if os.path.exists(filename):
        print("Found cache:", filename)
        vvecs = pk.load(open(filename, 'rb'))
    else:
        print("Cache not found:", filename)
        actions = phyre_gen_candidates(task_id, PHYRE_SET)
        BATCH_SIZE = 500
        vvecs = {}
        jj = 0
        for i in range(BATCH_SIZE, MAX_TEST_ACTIONS + BATCH_SIZE, BATCH_SIZE):
            batch = actions[i-BATCH_SIZE:i]
            if len(batch) < 1:
                break
            pred = inference.predict(batch)
            vvec = get_velocity_vector_from_bboxes_phyre(simulator, batch, pred['boxes'], pred['objects'])
            for vec in vvec:
                vvecs[tuple(x.item() for x in actions[jj])] =  vec
                jj += 1
        pk.dump(vvecs, open(filename, "wb"))
    return vvecs

def find_closest_candidate(candidates, candidate):
    tx, ty, tz = candidate
    return min(
        candidates,
        key=lambda t: (tx - t[0])**2 + (ty - t[1])**2 + (tz - t[2])**2
    )
    
def phyre_gen_candidates(task_id, PHYRE_SET):
    if PHYRE_SET != "2B":
        cache = get_default_100k_cache('ball')
    else:
        cache = get_default_100k_cache('two_balls')
    actions = cache.action_array[:MAX_TEST_ACTIONS].astype(np.double) # get 10K actions
    # filter unfeasible actions for this puzzle
    statuses = cache.load_simulation_states(task_id)[:MAX_TEST_ACTIONS]
    print("Actions with solution: ", len(actions[statuses==1]))
    actions = actions[statuses != 0]
    return torch.tensor(actions, dtype=torch.double)

def visualizePathSingleImagePhyre(simulator, action, path_dict, images):
    _, ax = plt.subplots()
    initial = observations_to_float_rgb(images[0])
    final = observations_to_float_rgb(images[-1])
    white_mask = np.all(final == 1.0, axis=2)
    # add alpha
    final = np.dstack((final, np.ones(final.shape[:2], dtype=float)))
    final[white_mask,3] = 0

    colors = {}
    plt.imshow(initial, alpha=1)
    plt.imshow(initial, alpha=0.4)
    plt.imshow(final)
    for object, path in path_dict.items():
        prev_center = path[0][0], 256-path[0][1]
        if object not in colors:
            colors[object] = initial[int(prev_center[1]), int(prev_center[0])]
        for step in path[1:]:
            color = colors[object]
            center = step[0], 256-step[1]
            ax.plot([prev_center[0], center[0]], [prev_center[1], center[1]], color=color, linewidth=1.3)
            prev_center = center
    plt.xticks([])
    plt.yticks([])
    plt.savefig('puzzle.svg', dpi=300)
    plt.show()
    