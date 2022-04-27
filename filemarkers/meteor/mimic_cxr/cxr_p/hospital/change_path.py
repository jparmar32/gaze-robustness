import pickle
import os


with open('test_list.pkl', 'rb') as f:
    data = pickle.load(f)
    
'''new_data = []

for i in data:

    path = i[0]
    new_path = path[:4] + "/data" + path[4:]
    new_data.append((new_path, i[1]))

with open("./test_list.pkl","wb") as pkl_f:
    pickle.dump(new_data,pkl_f)'''