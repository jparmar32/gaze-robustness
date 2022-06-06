from turtle import pos
import numpy as np 
import matplotlib.pyplot as plt
import pickle


positive_image_train = np.load('./data/synth_hard/train_img_260.npy')
positive_image_val = np.load('./data/synth_hard/val_img_100.npy')
positive_image_test = np.load('./data/synth_hard/test_img_100.npy')

print(np.unique(positive_image_train))
plt.imshow(positive_image_train, cmap='gray')
plt.savefig("./positive_image_train.png")
plt.close()

plt.imshow(positive_image_val, cmap='gray')
plt.savefig("./positive_image_val.png")
plt.close()

plt.imshow(positive_image_test, cmap='gray')
plt.savefig("./positive_image_test.png")
plt.close()

print(f"train_img_size: {positive_image_train.shape}")
print(f"val_img_size: {positive_image_val.shape}")
print(f"test_img_size: {positive_image_test.shape}")

negative_image_train = np.load('./data/synth_hard/train_img_50.npy')
negative_image_val = np.load('./data/synth_hard/val_img_50.npy')
negative_image_test = np.load('./data/synth_hard/test_img_50.npy')

plt.imshow(negative_image_train, cmap='gray')
plt.savefig("./negative_image_train.png")
plt.close()

plt.imshow(negative_image_val, cmap='gray')
plt.savefig("./negative_image_val.png")
plt.close()

plt.imshow(negative_image_test, cmap='gray')
plt.savefig("./negative_image_test.png")
plt.close()

positive_seg_train = np.load('./data/synth_hard/train_seg_260.npy')
positive_seg_val = np.load('./data/synth_hard/val_seg_100.npy')
positive_seg_test = np.load('./data/synth_hard/test_seg_100.npy')

plt.imshow(positive_seg_train, cmap='gray')
plt.savefig("./positive_seg_train.png")
plt.close()

plt.imshow(positive_seg_val, cmap='gray')
plt.savefig("./positive_seg_val.png")
plt.close()

plt.imshow(positive_seg_test, cmap='gray')
plt.savefig("./positive_seg_test.png")
plt.close()

print(f"train_seg_size: {positive_seg_train.shape}")
print(f"val_seg_size: {positive_seg_val.shape}")
print(f"test_seg_size: {positive_seg_test.shape}")

negative_seg_train = np.load('./data/synth_hard/train_seg_50.npy')
negative_seg_val = np.load('./data/synth_hard/val_seg_50.npy')
negative_seg_test = np.load('./data/synth_hard/test_seg_50.npy')

plt.imshow(negative_seg_train, cmap='gray')
plt.savefig("./negative_seg_train.png")
plt.close()

plt.imshow(negative_seg_val, cmap='gray')
plt.savefig("./negative_seg_val.png")
plt.close()

plt.imshow(negative_seg_test, cmap='gray')
plt.savefig("./negative_seg_test.png")
plt.close()


with open(f"./data/synth_hard/test_labels.pkl", "rb") as f:
    img_labels = pickle.load(f)
