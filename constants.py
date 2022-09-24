import numpy as np

K = [
    6.5308391993466671e002,
    0.0,
    3.1950000000000000e002,
    0.0,
    6.5308391993466671e002,
    2.3950000000000000e002,
    0.0,
    0.0,
    1.0,
]
D = [
    7.0834633684407095e-002,
    6.9140193737175351e-002,
    0.0,
    0.0,
    -1.3073460323689292e000,
]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32(
    [
        [6.825897, 6.760612, 4.402142],
        [1.330353, 7.122144, 6.903745],
        [-1.330353, 7.122144, 6.903745],
        [-6.825897, 6.760612, 4.402142],
        [5.311432, 5.485328, 3.987654],
        [1.789930, 5.393625, 4.413414],
        [-1.789930, 5.393625, 4.413414],
        [-5.311432, 5.485328, 3.987654],
        [2.005628, 1.409845, 6.165652],
        [-2.005628, 1.409845, 6.165652],
        [2.774015, -2.080775, 5.048531],
        [-2.774015, -2.080775, 5.048531],
        [0.000000, -3.116408, 6.097667],
        [0.000000, -7.415691, 4.070434],
    ]
)

reprojectsrc = 0.5 * np.float32(
    [
        [20.0, 20.0, 20.0],
        [10.0, 10.0, -10.0],
        [10.0, -10.0, -10.0],
        [20.0, -20.0, 20.0],
        [-20.0, 20.0, 20.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, -10.0],
        [-20.0, -20.0, 20.0],
    ]
)

line_pairs = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
