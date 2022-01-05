import shutil
import pickle
import os 

dir_list = "./mimic_cxr/cxr_p/hospital/test_list.pkl"
to_location = "/media/ood_data/mimic_cxr/cxr_p/hospital"

with open(dir_list, 'rb') as f:
    data = pickle.load(f)

for i in data:
    #print(i[0])
    '''path = i[0].split("/")
    path_new = "/".join(path[-3:-1])
    new_to_location = os.path.join(to_location, path_new)
    try:
        os.makedirs(new_to_location)
    except:
        pass
    shutil.copy(i[0], new_to_location)'''
    #print(to_location + i[0][:])
    shutil.copy(i[0], to_location)

#path = os.path.join(parent_dir, directory)
  
# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
#os.mkdir(path)