import torch
import h5py

vlist_2_prefix = {
    'bbt_frames':'',
    'castle_frames':'castle_',
    'met_frames':'met_',
    'grey_frames':'grey_',
    'friends_frames':'friends_',
    'house_frames':'house_'
}

old = h5py.File('/home/jumperkables/kable_management/data/tvqa/motion_features/tvqa_c3d_fc6_features.hdf5', 'r', driver=None)
new = h5py.File('/home/jumperkables/kable_management/data/tvqa/motion_features/fixed_tvqa_c3d_fc6_features.hdf5', 'w', driver=None)


for idx, clip in enumerate(old.keys()):
    if clip[0] != 's':
        clip_split = clip.split('_')[1:]
        clip_new = '_'.join(clip_split)
        new.create_dataset(clip_new, data=old[clip])
    else:
        new.create_dataset(clip, data=old[clip])
    print(idx)

old.close()
new.close()
