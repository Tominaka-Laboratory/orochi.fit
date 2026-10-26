par = {
    "sampler": {"value": ["TPE"], "member": ["TPE"]},
    "directions": {"value": ["minimize"]},
    "n_trials": {"value": [100]},
    "n_jobs": {"value": [1]},
    "continue_study": {"value": False},
    "dashboard_url":{"value": r"http://127.0.0.1:8080/"},

    # "database_dir": {"value": ""},
    "database_type": {
        "value": ["journal"],
        "member": ["journl","sqlite", "mysql+pymysql"],
    },
    "database_user": {"value": ["root"]},
    "database_password": {"value": []},
    "database_host": {"value": ["localhost"]},
    "database_port": {"value": [3306]},
    # "study_name": {"value": ["test"]},

    "save_function": {"value": [None]},
    # "score_function": {"value": [None]},
    "early_stop": {"value": [None]},
    "logging_callback": {"value": [None]},
}

par_key_info = {
    "sampler": {"value": [str, None, 1]},
    "directions": {"value": [str, None, -1]},
    "n_trials": {"value": [int, None, 1]},
    "n_jobs": {"value": [int, None, 1]},
    "continue_study": {"value": [bool, None, 1]},
    "dashboard_url": {"value": [str, None, 1]},

    # "database_dir": {"value": [str]},
    "database_type": {"value": [str]},
    "database_user": {"value": [str]},
    "database_password": {"value": [str]},
    "database_port": {"value": [str]},
    "database_host": {"value": [str]},
    # "study_name": {"value": [str, None, 1]},

    "save_function": {"value": [object]},
    # "score_function": {"value": [object, None, 1]},
    "early_stop": {"value": [object, None, 1]},
    "logging_callback": {"value": [object, None, 1]},

}
