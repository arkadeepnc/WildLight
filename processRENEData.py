import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
from glob import glob
import imageio
from skimage import img_as_ubyte
from tqdm import tqdm


def parse_RENE_dataset(path):
        ''' parses the RENE datasets https://github.com/eyecan-ai/rene'''
        light_fnames = glob(path+'/*', recursive = False)
                # light_locations = []
        # taking 3 frames from each of 39 light configs
        image_idxs = np.arange(49)
        global_counter = 37 * 3
        cam_poses = []
        light_poses = []
        cam_intrinsics = []
        images = []
        _light_config_counter = 0
        while global_counter >=1:
            np.random.shuffle(image_idxs)
            light_loc = np.loadtxt(light_fnames[_light_config_counter]+'/light.txt').reshape(4,4)
            with open(light_fnames[_light_config_counter]+'/camera.yaml','r') as file:
                cam_params = yaml.safe_load(file)
            _per_file_ctr = 0
            while _per_file_ctr < 10 :
                # print(_light_config_counter,'<?????')
                # print(image_idxs[_per_file_ctr],'<-------')
                image_name = light_fnames[_light_config_counter] + '/data/' + f"{image_idxs[_per_file_ctr]:02d}"+ '_image.png'
                image_arr = cv2.imread(image_name)
                pose_file_name = light_fnames[_light_config_counter] + '/data/' + f"{image_idxs[_per_file_ctr]:02d}"+ '_pose.txt'
                camera_pose = np.loadtxt(pose_file_name).reshape(4,4)
                if image_arr.sum() == 0:
                    print('loaded empty image, skipping --> {}'.format(image_name))
                    np.random.shuffle(image_idxs)
                    continue


                cam_poses.append(camera_pose) # _string = 'cam_pose' + str(global_counter)
                light_poses.append(light_loc) #  'light_pose' + str(global_counter)
                cam_intrinsics.append(np.asarray(cam_params['intrinsics']['camera_matrix']))
        
                
                # cam_light_poses[light_pose_string] = light_loc
                # cam_light_poses[cam_pose_string] = camera_pose
                # cv.imshow('image',image_arr)
                # cv.waitKey(0)
                images.append((image_arr/255.).astype(np.float32))
                _per_file_ctr += 1
                global_counter -= 1
            _light_config_counter += 1
        # cv.destroyAllWindows()
        return images, cam_poses, light_poses, cam_intrinsics

def reflectanceFromRENE(path):
    light_fnames = glob(path+'/*', recursive = False)
    # light_locations = []
    # taking 3 frames from each of 39 light configs
    camera_posed_image_lists = []
    for light_fname in light_fnames:
        image_fnames = glob(light_fname + '/data/' + '*.png')
        camera_posed_image_lists.append(image_fnames)
    
    for camera_posed_image_list in camera_posed_image_lists:
        # print(len(camera_posed_image_list),'<????')
        assert len(camera_posed_image_list) == 50
    
    # writer_col = imageio.mimwrite('./' + 'reflectances.gif', mode='I', duration=1.0)
    
    # read image of the same pose
    # specularities = []
    pose_idxs = np.arange(50).tolist()
    for pose_idx in tqdm(pose_idxs):
        illum_imgs_at_pose = []
        for light_fname in light_fnames:
            image_name = light_fname + '/data/' + f"{pose_idx:02d}"+ '_image.png'
            image_arr = cv2.imread(image_name)
            image_arr_gs = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
            if image_arr_gs.sum() == 0:
                print('loaded empty image, skipping --> {}'.format(image_name))
                continue
            illum_imgs_at_pose.append(image_arr_gs)

        # for img in illum_imgs_at_pose:
        #     cv2.imshow('zds',img)
        #     cv2.waitKey(5)
        # cv2.destroyAllWindows()
        if illum_imgs_at_pose:
            idst_var = estIdt(illum_imgs_at_pose, string = None )
            specularities = img_as_ubyte(idst_var)
            plt.imsave('./specularities_' + str(pose_idx) + '.png', specularities, cmap = 'gray' )
            # writer_col.append_data(img_as_ubyte(idst_var))

    
    
    # imageio.mimwrite('./' + 'specularities.gif', specularities ,  mode='I', duration=0.8)

        # plt.imshow(idst_var, cmap = 'gray')
        # plt.show()
        # image_gs_var = np.var(np.dstack(illum_imgs_at_pose), axis = -1)
        # plt.imshow(image_gs_var, cmap = 'gray')
        # plt.show()
    # writer_col.close()



    # image_idxs = np.arange(49)


    # global_counter = 37 * 48
    # cam_poses = []
    # light_poses = []
    # cam_intrinsics = []
    # images = []
    
    # _light_config_counter = 0
    # while global_counter >=1:
    #     # np.random.shuffle(image_idxs)
    #     # light_loc = np.loadtxt(light_fnames[_light_config_counter]+'/light.txt').reshape(4,4)
    #     # with open(light_fnames[_light_config_counter]+'/camera.yaml','r') as file:
    #     image_gs = []
    #     _per_file_ctr = 0
    #     while _per_file_ctr < 48:
    #          image_name = light_fnames[_light_config_counter] + '/data/' + f"{image_idxs[_per_file_ctr]:02d}"+ '_image.png'
    #          image_arr = cv2.imread(image_name)
    #          image_arr_gs = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    #          if image_arr_gs.sum() == 0:
    #                 print('loaded empty image, skipping --> {}'.format(image_name))
    #                 # np.random.shuffle(image_idxs)
    #                 _per_file_ctr += 1
    #                 continue
    #          image_gs.append(image_arr_gs)
    #          _per_file_ctr += 1
    #          global_counter -= 1


        
    #     image_gs_tensor = np.dstack(image_gs)
    #     # image_gs_var = np.var(image_gs_tensor, axis = -1)
    #     idst_var = estIdt(image_gs_tensor, string = str(_light_config_counter) )
    #     plt.imshow(idst_var, cmap = 'gray')
    #     plt.show()
    #     exit()
    # _light_config_counter += 1

def _scale(x):
    out_range = [0., 1.]
    domain = np.min(x.flatten()), np.max(x.flatten())
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2    

def estIdt(image_list, fem_d = 1, string = None):

    observations = np.dstack(image_list[0:-1])
    # assert observations.shape[-1] == 9 # all 9 channels and one special frame
   
    assert len(observations.shape) == 3 # W,H,channel all grayscale

    if observations.dtype == 'uint16':
        observations = observations/65535.0
    elif observations.dtype == 'uint8':
        observations = observations/255.0

    observations_dt = np.zeros(observations.shape)
    nLED = observations.shape[2]
    for iled in range(nLED):
        # jled_ = np.int(iled) - 1 # previous LED
        # kled_ = np.int(iled) + 1 # next LED
        jled_ = int(iled) - fem_d # previous LED
        kled_ = int(iled) + fem_d # next LED
        if jled_ < 0:
            jled_ = (jled_ + nLED) 
        if kled_ >= nLED:
            # kled_ = 0
            kled_ =  kled_ % nLED
        observations_dt[:,:,iled] = \
            (observations[:,:,kled_] - observations[:,:,jled_]) / float(fem_d*2)  
    _var = np.var(observations, axis = -1)
    # plt.imshow(_var)
    # plt.imshow((_var > np.percentile(_var, 99.)))
    # plt.show()

    if string:        
        writer_col = imageio.get_writer('./' + string + 'ratio-images.gif', mode='I', duration=0.8)
        for i in range(nLED):
            # plt.imshow(observations_dt[:,:,i])
            # plt.show()
            _frame = img_as_ubyte(_scale(observations_dt[:,:,i]))
            writer_col.append_data(_frame)
        writer_col.close()
    # return _var > np.percentile(_var, 99.)
    # return _var/_var.max()
    return _scale(_var)

if __name__ == "__main__":
     reflectanceFromRENE('/home/arkadeepchaudhury/Downloads/kittens_public')
     

        

     