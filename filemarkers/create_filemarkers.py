import pickle
import csv 
import os 

##both cxr-a and cxr-p
def create_chexpert_filemarker(dset,shift):
    base_dir = "/media/chexpert/"
    if shift == 'hospital':

        if dset == "a":
            test_file = "/home/jsparmar/test_data/CXR-A Splits/hospital_mean_55.395_shift_chexpert_test_cxra.csv"
            save_dir = "./chexpert/cxr_a/hospital"
        else:
            test_file = "/home/jsparmar/test_data/CXR-P Splits/hospital_mean_51.126_shift_chexpert_test_cxrp.csv"
            save_dir = "./chexpert/cxr_p/hospital"
    else:
        if dset == "a":
            test_file = "/home/jsparmar/test_data/CXR-A Splits/hospital_age_mean_89.602_shift_chexpert_test_cxra.csv"
            save_dir = "./chexpert/cxr_a/hospital_age"
        else:
            test_file = "/home/jsparmar/test_data/CXR-P Splits/hospital_age_mean_78.561_shift_chexpert_test_cxrp.csv"
            save_dir = "./chexpert/cxr_p/hospital_age"

    all_file_markers = []
    labs = []

    
    with open(test_file,"r") as cf:
        rows = csv.reader(cf)

        for i, row in enumerate(rows):
            if i == 0:
                print(row)
            if i > 0:

                val = row[0].split("/")
                image_path = '/'.join(val[1:])
                pth = os.path.join(base_dir,image_path)

                if row[1] != "No Finding":
                    lab = 1
                else:
                    lab = 0

                labs.append(lab)
                all_file_markers.append((pth, lab))

    test_dir = os.path.join(save_dir,"test_list.pkl")
    with open(test_dir,"wb") as pkl_f:
        pickle.dump(all_file_markers,pkl_f)


def chestxray8_helper(val, second_val):

    if val <= 1335:
        return "images_01"

    elif val <= 3923:

        if val == 3923 and second_val <= 13:
            
            return "images_02"
        elif val == 3923 and second_val > 13 :
            return "images_03"
        else:
            return "images_02"

    elif val <= 6585:

        if val == 6585 and second_val <= 6:
            
            return "images_03"
        elif val == 6585 and second_val > 6 :
            return "images_04"
        else:
            return "images_03"

    elif val <= 9232:

        if val == 9232 and second_val <= 3:
            
            return "images_04"
        elif val == 9232 and second_val > 3 :
            return "images_05"
        else:
            return "images_04"

    elif val <= 11558:

        if val == 11558 and second_val <= 7:
            
            return "images_05"
        elif val == 11558 and second_val > 7 :
            return "images_06"
        else:
            return "images_05"

    elif val <= 13774:

        if val == 13774 and second_val <= 25:
            
            return "images_06"
        elif val == 13774 and second_val > 25:
            return "images_07"
        else:
            return "images_06"

    elif val <= 16051:

        if val == 16051 and second_val <= 9:
            
            return "images_07"
        elif val == 16051 and second_val > 9:
            return "images_08"
        else:
            return "images_07"

    elif val <= 18387:

        if val == 18387 and second_val <= 34:
            
            return "images_08"
        elif val == 18387 and second_val > 34:
            return "images_09"
        else:
            return "images_08"

    elif val <= 20945:

        if val == 20945 and second_val <= 49:
            
            return "images_09"
        elif val == 20945 and second_val > 49:
            return "images_010"
        else:
            return "images_09"

    elif val <= 24717:

        if val == 24717 and second_val <= 0:
            
            return "images_010"
        elif val == 24717 and second_val > 0:
            return "images_011"
        else:
            return "images_010"

    elif val <= 28173:

        if val == 28173 and second_val <= 2:
            
            return "images_011"
        elif val == 28173 and second_val > 2:
            return "images_012"
        else:
            return "images_011"

    elif val <= 30805:
        return "images_012"
    
    return None

## cxr-a
def create_chestxray8_filemarker(dset= "a", shift = 'hospital'):
    

    base_dir = "/media/chestxray8/"

    if dset == "a":
        if shift == 'hospital':
            test_file = "/home/jsparmar/test_data/CXR-A Splits/hospital_shift_mean_52.313_chestxray8_test_cxra.csv"
            save_dir = "./chestxray8/cxr_a/hospital"
        else:
            test_file = '/home/jsparmar/test_data/CXR-A Splits/hospital_age_mean_62.45_shift_chestxray8_test_cxra.csv'
            save_dir = "./chestxray8/cxr_a/hospital_age"
    else:
        test_file = '/home/jsparmar/test_data/CXR-P Splits/hospital_age_mean_71.391_shift_chestxray8_test_cxrp.csv'
        save_dir = "./chestxray8/cxr_p/age"
        



    all_file_markers = []
    labs = []

    
    with open(test_file,"r") as cf:
        rows = csv.reader(cf)

        for i, row in enumerate(rows):
            if i == 0:
                print(row)
            if i > 0:

                val = row[0].split("_")
                val_one = int(val[0])
                vals = val[1].split('.')
                val_two = int(vals[0])
                file = chestxray8_helper(val_one, val_two)

                image_path = f"{file}/{row[0]}"
                pth = os.path.join(base_dir,image_path)

                if row[1] != "No Finding":
                    lab = 1
                else:
                    lab = 0

                labs.append(lab)
                all_file_markers.append((pth, lab))
    
    
    test_dir = os.path.join(save_dir,"test_list.pkl")
    with open(test_dir,"wb") as pkl_f:
        pickle.dump(all_file_markers,pkl_f)




    
## cxr-p
def create_mimic_cxr_filemarker():
    base_dir = "/mnt/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    test_file = "/home/jsparmar/test_data/CXR-P Splits/hospital_shift_mimicxr_test_cxrp.csv"
    all_file_markers = []
    labs = []

    
    with open(test_file,"r") as cf:
        rows = csv.reader(cf)

        for i, row in enumerate(rows):
            if i == 0:
                print(row)
            if i > 0:
                
                if int(row[1]) < 11000000:
                    pre_index = "p10"
                else: 
                    pre_index = "p11"

                image_path = f"{pre_index}/p{row[1]}/s{row[2]}/{row[0]}"
                pth = os.path.join(base_dir,image_path+".jpg")

                if row[3] == "Pneumothorax":
                    lab = 1
                else:
                    lab = 0

                labs.append(lab)
                all_file_markers.append((pth, lab))
    
    save_dir = "./mimic_cxr/cxr_p/hospital"
    test_dir = os.path.join(save_dir,"test_list.pkl")
    with open(test_dir,"wb") as pkl_f:
        pickle.dump(all_file_markers,pkl_f)




def main():
    create_chexpert_filemarker(dset="a", shift='hospital')
    create_chexpert_filemarker(dset="a", shift='hospital_age')
    create_chexpert_filemarker(dset="p", shift='hospital')
    create_chexpert_filemarker(dset="p", shift='hospital_age')

    create_chestxray8_filemarker(dset="a", shift='hospital')
    create_chestxray8_filemarker(dset="a", shift='hospital_age')
    create_chestxray8_filemarker(dset="p", shift='age')
    
    create_mimic_cxr_filemarker()
    

if __name__ == "__main__":
    main()
    