par = {
    "method": {
        "value": "least_squares",
        "member": [
            "leastsq",
            "least_squares"
        ],
    },
    "max_nfev": {"value": 1000},
    "iter_cb": {"value": None},
    "ftol": {"value": 1e-04},
    "xtol": {"value": 1e-04},
    "gtol": {"value": 1e-04},

    "diff_step": {"value": 1},
}

par_key_info = {  # key:{atribute:[type, index, shape]}
    "method": {
        "value": [str, None, 1],
        "members": [str, None, -1],
    },
    "max_nfev": {
        "value": [int, None, 1],
    },
    "iter_cb": {
        "value": [None, None, 1],
    },
    "ftol": {"value": [float, None, 1]},
    "xtol": {"value": [float, None, 1]},
    "gtol": {"value": [float, None, 1]},

    "diff_step": {"value": [float, None, 1]},

}
