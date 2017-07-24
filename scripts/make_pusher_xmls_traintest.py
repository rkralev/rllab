""" makes sure that pairs of objects are sampled from both sides """
from rllab.misc.mujoco_render import pusher
import glob
import random
import re

USE_DISTRACTORS_OR_TEXTURES = True
TABLE_CONSTANT = True
TWO_TEXTURES = False #True
base_dir = '/home/cfinn/code/rllab/'
docker_dir = '/root/code/rllab/'
objs = glob.glob(base_dir+'vendor/mujoco_models/*.stl')
obj_textures = glob.glob(base_dir+'vendor/textures/obj_textures/*.png')
table_textures = glob.glob(base_dir+'vendor/textures/table_textures/*.png')

if TWO_TEXTURES:
    obj_textures = [x for x in obj_textures if 'knitted_0079' in x or 'knitted_0076' in x]

# make distractor info first, to preserve random seed.
all_distr_objs = []
all_distr_objpaths = []
all_distr_scales = []
all_distr_masses = []
all_distr_damps = []

all_distr_textures = []
all_obj_textures = []
all_table_textures = []
prefix = ''

def get_object_stl(xml_file):
    with open(xml_file, 'r') as f:
        contents = f.read()
    obj = re.findall(r'/\S*\.stl',contents)[0]
    obj = '/home/cfinn' + obj[5:]
    scale = float(re.findall(r'mesh scale="[.0-9]*', contents)[0][12:])
    return obj, scale

if USE_DISTRACTORS_OR_TEXTURES:
    if TWO_TEXTURES:
        assert TABLE_CONSTANT
        prefix = 'train_ensure_texture2_woodtable_distractor_'
    elif TABLE_CONSTANT:
        prefix = 'train_ensure_woodtable_distractor_'  # TODO - need to code up texture matching in this case.
    else:
        prefix = 'train_ensure_distractor_'

    # first set of 1000 xml files
    #offset = 0
    #random.seed(14)
    # second set of 1000 xml files
    offset = 1000
    random.seed(15)

    for i in range(1000):
        if i % 2 == 0:
            other_i = i+1
            # even, look at next number
        else:
          other_i = i-1
        obj, scale = get_object_stl('/home/cfinn/code/rllab/vendor/mujoco_models/pusher' + str(other_i) + '.xml')
        all_distr_objs.append(obj)
        all_distr_objpaths.append( docker_path + obj[obj.index('vendor'):] )
        all_distr_scales.append(scale)
        #all_distr_objs.append(random.choice(objs))
        #all_distr_scales.append(random.uniform(0.5, 1.0))
        all_distr_masses.append(random.uniform(0.1, 2.0))
        all_distr_damps.append(random.uniform(0.2, 5.0))
        if i % 2 == 0:
            if TWO_TEXTURES:
                distr_id = random.randint(0,1)
                obj_id = 1-distr_id
            else:
                distr_id = random.randint(0, len(obj_textures)-1)
                obj_id = random.randint(0, len(obj_textures)-1)
        else:
            tmp = distr_id
            distr_id = obj_id
            obj_id = tmp
            #if TWO_TEXTURES:
            #    distr_id = 1-distr_id
        all_distr_textures.append(obj_textures[distr_id])
        all_obj_textures.append(obj_textures[obj_id])
        if TABLE_CONSTANT:
            table_texture = [tex for tex in table_textures if 'wpic_002' in tex][0]
            all_table_textures.append(table_texture)
        else:
            all_table_textures.append(random.choice(table_textures))



#i = 1
#if True:
for i in range(1000):
    random.seed(i+1+1000)
    random_obj = random.choice(objs)
    random_obj_dockerpath = docker_dir + random_obj[random_obj.index('vendor'):]
    random_scale = random.uniform(0.5, 1.0)
    random_mass = random.uniform(0.1, 2.0)
    random_damp = random.uniform(0.2, 5.0)
    if not USE_DISTRACTORS_OR_TEXTURES:
        pusher_model = pusher(mesh_file=random_obj,mesh_file_path=random_obj_dockerpath, obj_scale=random_scale,obj_mass=random_mass,obj_damping=random_damp)
        xml_file = base_dir + 'vendor/mujoco_models/pusher' + str(i) + '.xml'
        pusher_model.save(xml_file)

        pusher_model_local = pusher(mesh_file=random_obj,mesh_file_path=random_obj, obj_scale=random_scale,obj_mass=random_mass,obj_damping=random_damp)
        xml_file = base_dir + 'vendor/local_mujoco_models/pusher' + str(i) + '.xml'
        pusher_model_local.save(xml_file)
    else:
        pusher_model = pusher(mesh_file=random_obj,mesh_file_path=random_obj, obj_scale=random_scale,
                              obj_mass=random_mass,obj_damping=random_damp,
                              distractor_mesh_file=all_distr_objs[i], distractor_mesh_file_path=all_distr_objpaths[i],
                              actual_distr_scale=all_distr_scales[i],
                              distr_mass=all_distr_masses[i], distr_damping=all_distr_damps[i],
                              table_texture=all_table_textures[i], distractor_texture=all_distr_textures[i], obj_texture=all_obj_textures[i])
        xml_file = base_dir + 'vendor/local_mujoco_models/'+prefix+'pusher' + str(i+offset) + '.xml'
        pusher_model.save(xml_file)

        pusher_model_local = pusher(mesh_file=random_obj,mesh_file_path=random_obj, obj_scale=random_scale,
                                    obj_mass=random_mass,obj_damping=random_damp,
                                    distractor_mesh_file=all_distr_objs[i], actual_distr_scale=all_distr_scales[i],
                                    distr_mass=all_distr_masses[i], distr_damping=all_distr_damps[i],
                                    table_texture=all_table_textures[i], distractor_texture=all_distr_textures[i], obj_texture=all_obj_textures[i])
        xml_file = base_dir + 'vendor/local_mujoco_models/'+prefix+'pusher' + str(i+offset) + '.xml'
        pusher_model_local.save(xml_file)
