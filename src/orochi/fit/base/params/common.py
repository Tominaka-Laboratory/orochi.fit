par = {
    "obsx": {"value": []},
    "obsy": {"value": []},
    "simx_attr":{"value":"r"},
    "simy_attr":{"value":"Gr"},
    
    "fit_min": {"value": 1},
    "fit_max": {"value": 30},

    "x_extension": {"value": 0.0},
    "x_min_key": {"value": "rmin"},
    "x_max_key": {"value": "rmax"},

    "prefix": {"value": "mod"},
    "address_delimiter":{"value":"__"},

    "score_function":{"value":"Rw"},
    "score_type":{"value":"adj"},

}

par_key_info = {  # key:{atribute:[type, index, shape]}

    "obsx": {"value": [float, None, -1]},
    "obsy": {"value": [float, None, -1]},
    "simx_attr":{"value":[str, None, 1]},
    "simy_attr":{"value":[str, None, 1]},
    
    "fit_min": {"value": [float, None, 1]},
    "fit_max": {"value": [float, None, 1]},

    "x_extension": {"value": [float, None, 1]},
    "x_min_key": {"value": [str, None, 1]},
    "x_max_key": {"value": [str, None, 1]},

    "prefix": {"value": [str, None, 1]},
    "address_delimiter":{"value":[str, None, 1]},

    "score_function":{"value":[str, None, 1]},
    "score_type":{"value":[str, None, 1]}

}
