import pickle
import torch
import os
# 加载保存的数据
# with open('./daily_action_skeleton_data.pkl', 'rb') as file:
    # loaded_data = pickle.load(file)

def load_folders(directory):
    data = {}
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            folder_path = os.path.join(root, d)
            pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
            loaded_files = [torch.load(os.path.join(folder_path, f)) for f in pt_files]
            data[d] = loaded_files
    return data

directory_path = 'C:/Users/randulf/Desktop/data/extracted'  
loaded_data = load_folders(directory_path)

# view the shape of converted angle
def shape(data):
    for key, tensor_list in data.items():
        print(f"Key: {key}")
        for idx, tensor in enumerate(tensor_list):
            print(f"Tensor {idx + 1} shape: {tensor.shape}")

def cartesian_to_spherical(coords):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)

    spherical_coords = torch.stack((r, theta, phi), dim=-1)
    return spherical_coords

def cartisian_angle(tensor):
    '''计算每三个连续坐标构成的角度'''
    num = tensor.size(1)-2
    angles = torch.zeros(tensor.size(0), num)  # 创建一个用于存储结果的张量

    for i in range(num):  # 四个夹角
        vec1 = tensor[:, i, :] - tensor[:, i+1, :]
        vec2 = tensor[:, i+1, :] - tensor[:, i+2, :]

        dot_product = (vec1 * vec2).sum(dim=1)
        norm1 = torch.norm(vec1, dim=1)
        norm2 = torch.norm(vec2, dim=1)
        cos_angle = dot_product / (norm1 * norm2)
        angles[:, i] = torch.acos(cos_angle.clamp(-1, 1))

    return angles

def spherical_tensor(tensor):
    # 获取数据形状
    batch_size, num_points, _ = tensor.size()

    # 创建一个与数据相同形状的结果张量
    result = torch.zeros(batch_size, num_points, 3)

    # 遍历每个元素并进行转换
    for i in range(num_points):
        x, y, z = tensor[:, i, 0], tensor[:, i, 1], tensor[:, i, 2]
        r, theta, phi = cartesian_to_spherical(x, y, z)
        result[:, i, 0] = r
        result[:, i, 1] = theta
        result[:, i, 2] = phi
    return result


# return the left arm as spherical data, where each joint is calculated on the coordinate of last joint
def extract_left_arm_arithmatic(original_data):
    keep_indices = [5, 6, 7, 8, 22] # 左臂
    # keep_indices = [9, 10, 11, 12, 24] # 右臂
    for key, tensor_list in original_data.items():
        for idx, tensor in enumerate(tensor_list):
            extracted = tensor[:,keep_indices,:]
            angle_arithmatic = torch.zeros(extracted.size(0), extracted.size(1))
            relative_pos = extracted[:,1,:] - extracted[:,0,:]
            spherical_coords = cartesian_to_spherical(relative_pos)
            angle_arithmatic[:,0] = spherical_coords[:,2] # 肩膀位置映射1：方位角
            angle_arithmatic[:,1] = spherical_coords[:,1] # 肩膀位置映射2：天顶角
            angles_left = cartisian_angle(extracted) # 其他关节角
            angle_arithmatic[:,2:] = angles_left
            
            # 用0填充angle6
            zeros = torch.zeros(angle_arithmatic.size(0),1)
            angle_arithmatic = torch.cat((angle_arithmatic, zeros), dim=1)
            
            original_data[key][idx] = angle_arithmatic
    return original_data

spherical_data = extract_left_arm_arithmatic(loaded_data)

# 如果需要`.npy`格式的数据
# for key in spherical_data.keys():
#     spherical_data[key] = [tensor.numpy() for tensor in spherical_data[key]]

with open('ntu-60-extracted.pkl', "wb") as file:
    pickle.dump(spherical_data, file)