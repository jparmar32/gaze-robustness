import os
import shutil
import pickle

import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
import pydicom

POS_FEEDBACK_COLOR = np.array([0, 169, 255])
NEG_FEEDBACK_COLOR = np.array([255, 64, 64])


class FeedbackInterface:
    def __init__(
        self,
        img_ids: list,
        data_dir: str,
        size: tuple = (224, 224),
        num_examples: int = 2,
        rank_by: list = None,
        save_dir: str = None,
    ):
        """A class representing a Gradio interface for providing feedback on slices.
        Args:
            img_ids (list, required): A list of strings to the images that would like to be annotated.
            data_dir (str, required): Folder in which img_ids are stored.
            size (tuple, optional): Images will be reshaped to this size when displayed
                in gradio. Defaults to (224, 224).
            num_examples (int, optional): The number of examples to show in the
                interface. Defaults to 2.
            rank_by (list, optional): Indices by which to present examples from img_paths.
            save_dir (str, optional): A directory where the feedback will be saved.
                Defaults to None, in which case a directory named "feedback" will be
                created and used in the current working directory.
        """
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            self.save_dir = "feedback"

        self.images_save_dir = os.path.join(self.save_dir, "images")
        self.annotations_save_dir = os.path.join(self.save_dir, "annotations")

        self.data_dir = data_dir
        self.img_ids = img_ids
        self.size = size
        self.num_examples = num_examples
        
        os.makedirs(self.images_save_dir , exist_ok=True)
        os.makedirs(self.annotations_save_dir , exist_ok=True)
        print(f"The submitted feedback will be saved at {self.annotations_save_dir}")

        # prepare examples
        if rank_by:
            example_idxs = rank_by[:num_examples]
        else:
            example_idxs = np.random.choice(np.arange(len(self.img_ids)), size=self.num_examples, replace=False)

      
        examples = []

        transform = transforms.Compose([transforms.Resize(size)])

        for rank, example_idx in enumerate(example_idxs):
            example_idx = int(example_idx)
            img_id = self.img_ids[example_idx]
            img_path = os.path.join(self.data_dir, img_id)

            if img_path[-3:] == "jpg":
                img = Image.open(img_path)
            elif img_path[-3:] == "dcm":
                ds = pydicom.dcmread(img_path)
                img = ds.pixel_array
                img = Image.fromarray(np.uint8(img))
            else:
                raise ValueError("This image type is not yet supported")

            img_id = img_id[:-4]

            image = transform(img)

            image_path = os.path.join(self.images_save_dir, f"{img_id}.jpg")
            image.save(image_path)

            examples.append(
                [
                    rank,
                    example_idx,
                    img_id,
                    image_path,
                    None,
                ]
            )

        # define feedback function
        def submit_feedback(rank, example_idx, img_id, img, feedback_label):

            '''original_img_path = os.path.join(self.images_save_dir, f"{img_id}.jpg")
            original_image = np.array(Image.open(original_img_path))
            mask = img - original_image
            '''

            mask = np.zeros_like(img).astype(int)
            mask[(img == POS_FEEDBACK_COLOR).all(axis=-1)] = 1
            mask[(img == NEG_FEEDBACK_COLOR).all(axis=-1)] = -1
            mask = np.squeeze(mask[:,:,0])
            np.save(os.path.join(self.annotations_save_dir, f"{img_id}_lungmask.npy"), mask)


            np.save(os.path.join(self.annotations_save_dir, f"{img_id}_lungmask.npy"), mask)
            return []

        iface = gr.Interface(
            submit_feedback,
            [
                gr.inputs.Number(),
                gr.inputs.Number(),
                gr.inputs.Textbox(),
                gr.inputs.Image(),
                gr.inputs.Radio(choices=["positive", "negative", "abstain"]),
            ],
            outputs=[],
            examples=examples,
            layout="vertical",
        )
        self.interface = iface.launch(inbrowser=False, inline=False)

    def __del__(self):
        shutil.rmtree(self.imgs_dir)


if __name__ == "__main__":

    

    file_dir = os.path.join("./filemarkers", "cxr_p")
    file_markers_dir = os.path.join(file_dir, "trainval_list_gold.pkl") #gold
        
    with open(file_markers_dir, "rb") as fp:
        file_markers = pickle.load(fp)

    img_ids = [file_marker[0] for file_marker in file_markers]
    data_dir = os.path.join("/media", "pneumothorax/dicom_images")
    save_dir = 'lung_segmentations'

    if not os.path.isdir(os.path.join(save_dir,'annotations')):
        FeedbackInterface(img_ids=file_markers,data_dir=data_dir, num_examples=20, save_dir=save_dir)
    else:
        
        segmented_ids = []
        for segmentation in os.listdir(os.path.join(save_dir,'annotations')):
            segmented_ids.append(segmentation[:-1*(len("_lungmask.npy"))])

        rank_by = [i for i in range(len(file_markers))]
        for segmented_id in segmented_ids:
            to_rem_idx = file_markers.index(segmented_id)
            rank_by.remove(to_rem_idx)

        num_examples = min(20, len(rank_by))

        if num_examples != 0:
            FeedbackInterface(img_ids=file_markers,data_dir=data_dir, num_examples=20, rank_by=rank_by, save_dir=save_dir)

        

   



