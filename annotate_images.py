import os
import shutil
import time
import uuid

import gradio as gr
import numpy as np
import PIL
from meerkat import DataPanel, ListColumn, NumpyArrayColumn

POS_FEEDBACK_COLOR = np.array([0, 169, 255])
NEG_FEEDBACK_COLOR = np.array([255, 64, 64])


class FeedbackInterface:
    def __init__(
        self,
        dp: DataPanel,
        img_column: str = "img",
        label_column: str = "label",
        id_column: str = "image_id",
        size: tuple = (224, 224),
        rank_by: str = None,
        num_examples: int = 100,
        save_dir: str = None,
    ):
        """A class representing a Gradio interface for providing feedback on slices.
        Args:
            dp (DataPanel): the datapanel with examples to provide feedback on
            img_column (str, optional): The column in dp containing the images. Must
                materialize to a PIL image.  Defaults to "img".
            label_column (str, optional): The label of the image, to be displayed
                alongside the image in the interface. Defaults to "label".
            id_column (str, optional): A column that uniquely identifies each image in
                the DataPanel. This is the only information that will be saved alongside
                the feedback labels and masks. You should pick the column such that a
                simple `ms.merge` with the original `dp`, will give you the labels
                alongside the original imags. Defaults to "image_id".
            size (tuple, optional): Images will be reshaped to this size when displayed
                in gradio. Defaults to (224, 224).
            rank_by (str, optional): A column in `dp` containing scalar values by which
                the examples in `dp` should be sorted when . Note only the top
                `num_examples` will be shown. Defaults to None, in which case the
                examples are ordered randomly.
            num_examples (int, optional): The number of examples to show in the
                interface. Defaults to 100.
            save_dir (str, optional): A directory where the feedback will be saved.
                Defaults to None, in which case a directory named "feedback" will be
                created and used in the current working directory.
        """
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            self.save_dir = "feedback"
        self.save_path = os.path.join(
            self.save_dir, time.strftime(f"fb_%y-%m-%d_%H-%M_{uuid.uuid1().hex[:6]}.dp")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(
            f"The submitted feedback will be saved as a `DataPanel` at {self.save_path}"
        )

        # prepare examples
        if rank_by is not None:
            example_idxs = (-dp[rank_by]).argsort()[:num_examples]
        else:
            example_idxs = np.random.choice(
                np.arange(len(dp)), size=num_examples, replace=False
            )

        self.imgs_dir = "_fb_imgs"
        os.makedirs(self.imgs_dir, exist_ok=True)
        examples = []
        for rank, example_idx in enumerate(example_idxs):
            example_idx = int(example_idx)
            label = dp[label_column][example_idx]
            image = dp[img_column][example_idx]
            image.resize(size)
            if not isinstance(image, PIL.Image.Image):
                raise ValueError("`img_column` must materialize to `PIL.Image`")
            image_path = os.path.join(self.imgs_dir, f"image_{example_idx}.jpg")
            image.save(image_path)

            examples.append(
                [
                    rank,
                    example_idx,
                    str(label),  # important this is a str, gradio hangs otherwise
                    image_path,
                    dp["feedback_label"][example_idx]
                    if "feedback_label" in dp
                    else None,
                ]
            )

        label_dp = dp[[id_column]].lz[example_idxs]
        label_dp["feedback_label"] = NumpyArrayColumn(["unlabeled"] * len(example_idxs))
        label_dp["feedback_mask"] = ListColumn([None] * len(example_idxs))

        self.label_dp = label_dp

        # define feedback function
        def submit_feedback(rank, example_idx, label, img, feedback_label):
            mask = np.zeros_like(img).astype(int)
            mask[(img == POS_FEEDBACK_COLOR).all(axis=-1)] = 1
            mask[(img == NEG_FEEDBACK_COLOR).all(axis=-1)] = -1
            self.label_dp["feedback_label"][rank] = feedback_label
            self.label_dp["feedback_mask"][rank] = mask
            self.label_dp.write(self.save_path)
            return []

        iface = gr.Interface(
            submit_feedback,
            [
                gr.inputs.Number(),
                gr.inputs.Number(),
                gr.inputs.Textbox(),
                gr.inputs.Image(shape=size),
                gr.inputs.Radio(choices=["positive", "negative", "abstain"]),
            ],
            outputs=[],
            examples=examples,
            layout="vertical",
        )
        self.interface = iface.launch(inbrowser=False, inline=False)

    def __del__(self):
        shutil.rmtree(self.imgs_dir)