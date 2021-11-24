import pathlib

image_id_list = list(pathlib.Path("../../dataset/labelled/labelled_black_caps").glob("*.jpg"))
label_list = list(pathlib.Path("../../dataset/labelled/labelled_black_caps").glob("*.npy"))


def rename(file):
    file.rename(pathlib.Path(file.parent, "black_" + file.stem + file.suffix))


for file in image_id_list:
    rename(file)


for file in label_list:
    rename(file)
