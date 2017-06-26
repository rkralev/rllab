from rllab.misc.mujoco_render import pusher
import glob
import random

USE_DISTRACTORS = True
base_dir = '/home/cfinn/code/rllab/'
docker_dir = '/root/code/rllab/'
objs = glob.glob(base_dir+'vendor/mujoco_models/*.stl')
obj_textures = glob.glob(base_dir+'vendor/textures/obj_textures/*.png')
table_textures = glob.glob(base_dir+'vendor/textures/table_textures/*.png')

# make distractor info first, to preserve random seed.
all_distr_objs = []
all_distr_scales = []
all_distr_masses = []
all_distr_damps = []

all_distr_textures = []
all_obj_textures = []
all_table_textures = []
if USE_DISTRACTORS:
    prefix = 'distractor_'
    random.seed(14)
    for i in range(1000):
        all_distr_objs.append(random.choice(objs))
        all_distr_scales.append(random.uniform(0.5, 1.0))
        all_distr_masses.append(random.uniform(0.1, 2.0))
        all_distr_damps.append(random.uniform(0.2, 5.0))
        all_distr_textures.append(random.choice(obj_textures))
        all_obj_textures.append(random.choice(obj_textures))
        all_table_textures.append(random.choice(table_textures))


#i = 1
#if True:
for i in range(1000):
    random.seed(i+1+1000)
    random_obj = random.choice(objs)
    random_obj_path = docker_dir + random_obj[random_obj.index('vendor'):]
    random_scale = random.uniform(0.5, 1.0)
    random_mass = random.uniform(0.1, 2.0)
    random_damp = random.uniform(0.2, 5.0)
    if not USE_DISTRACTORS:
        pusher_model = pusher(mesh_file=random_obj,mesh_file_path=random_obj_path, obj_scale=random_scale,obj_mass=random_mass,obj_damping=random_damp)
        xml_file = base_dir + 'vendor/mujoco_models/pusher' + str(i) + '.xml'
        pusher_model.save(xml_file)

        pusher_model_local = pusher(mesh_file=random_obj,mesh_file_path=random_obj, obj_scale=random_scale,obj_mass=random_mass,obj_damping=random_damp)
        xml_file = base_dir + 'vendor/local_mujoco_models/pusher' + str(i) + '.xml'
        pusher_model_local.save(xml_file)
    else:
        pusher_model_local = pusher(mesh_file=random_obj,mesh_file_path=random_obj, obj_scale=random_scale,
                                    obj_mass=random_mass,obj_damping=random_damp,
                                    distractor_mesh_file=all_distr_objs[i], distr_scale=all_distr_scales[i],
                                    distr_mass=all_distr_masses[i], distr_damping=all_distr_damps[i],
                                    table_texture=all_table_textures[i], distractor_texture=all_distr_textures[i], obj_texture=all_obj_textures[i])
        xml_file = base_dir + 'vendor/local_mujoco_models/'+prefix+'pusher' + str(i) + '.xml'
        pusher_model_local.save(xml_file)
