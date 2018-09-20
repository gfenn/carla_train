import numpy as np
import cv2

BASE_CLASSIFICATIONS = 13
KEEP_CLASSES = [
    [0],  # 0 -> None
    [1, 2, 3, 4, 5, 9, 10, 11, 12],  # 1 -> Misc
    [6],  # 2 -> Road lines
    [7],  # 3 -> Roads
    [8]  # 4 -> Sidewalks
]
KEEP_CLASSIFICATIONS = len(KEEP_CLASSES)

CLASS_CONVERTER = {}
for idx, sublist in enumerate(KEEP_CLASSES):
    for subclass in sublist:
        CLASS_CONVERTER[subclass] = idx


# TODO -- Optimize this
def reduce_classifications(classifications):
    # Start out with classifications as a 2d array of INT values
    # Convert into a 2d array of INT values (with reduced classes set)
    new_classes = np.array(classifications)
    for x in range(classifications.shape[0]):
        for y in range(classifications.shape[1]):
            new_classes[x, y] = CLASS_CONVERTER[classifications[x, y]]
    return new_classes


def resize_classifications(classifications, new_size):
    # Convert into 3d
    base_shape_expanded = (KEEP_CLASSIFICATIONS, classifications.shape[0], classifications.shape[1])
    classes_cube_start = np.zeros(base_shape_expanded, dtype=np.float32)
    for x in range(classifications.shape[0]):
        for y in range(classifications.shape[1]):
            value = classifications[x, y]
            classes_cube_start[value, x, y] = 1.0

    # Resize down
    new_size_deep = (KEEP_CLASSIFICATIONS, new_size[0], new_size[1])
    new_resized_deep = np.zeros(new_size_deep)
    for idx in range(0, KEEP_CLASSIFICATIONS):
        new_resized_deep[idx] = cv2.resize(classes_cube_start[idx], (new_size[1], new_size[0]))

    # Argmax to flatten back into classifications
    return np.argmax(new_resized_deep, axis=0)

def fuse_with_depth(classifications, depth, extra_layers=0):
    # Determine shapes
    shape = classifications.shape
    fused_shape = (shape[0], shape[1], KEEP_CLASSIFICATIONS + extra_layers)

    # Make a base obs where everything is 1 (aka max distance away)
    obs = np.full(fused_shape, 1, dtype=np.float32)
    for x in range(shape[0]):
        for y in range(shape[1]):
            classification = classifications[x, y]
            d = depth[x, y]
            obs[x, y, classification] = depth[x, y]
    return obs

