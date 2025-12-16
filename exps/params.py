# Parameters for benchmarking experiments
params = {
        "general": {
            "model_name": "c2f_tcn",  # str; model name (run project.help("model") for more info)
            "metric_functions": {"precision", "recall", "f1"},  # set; set of metric names (run project.help("metrics") for more info)
            "ignored_clips": None,  # list; a list of string clip ids (agent names) to be ignored
            "len_segment": 1028,  # int; the length of segments (in frames) to cut the videos into
            "overlap": 0.75,  # float; the overlap (in frames) between neighboring segments
            "interactive": False,  # bool; if true, annotations are assigned and features are computed for pairs of clips (animals)
            "exclusive": False,  # bool; if true, single-label classification is used; otherwise multi-label
        },
        "data": {
            "data_suffix": "_pose3d_smpl.h5",  # set; the data files should have the format of {video_id}{data_suffix}
            "feature_suffix": None,  # str; the feature files should be stored in the data folder and named {video_id}{feature_suffix}
            "annotation_suffix": "_labels.pickle",  # str | set, the annotation files are named {video_id}{annotation_suffix}
            "canvas_shape": [100, 100, 100],  # list; the size of the canvas where the pose was defined
            "ignored_bodyparts": None,  # set; the set of string names of bodyparts to ignore
            "likelihood_threshold": 0,  # float; coordinates with lower likelihood values will be ignored
            "behaviors": None,  # set; the behaviors to predict (if null, it will be inferred from the data)
            "filter_annotated": False,  # bool; discard long unannotated intervals during training
            "filter_background": False,  # bool; only label frames as background if a behavior is annotated somewhere close
            "visibility_min_score": 0,  # float; the minimum visibility score for visibility filtering
            "visibility_min_frac": 0,  # float; the minimum fraction of visible frames for visibility filtering
        },
        "training": {
            "lr": 0.001,  # float; learning rate
            "device": "auto",  # str; device
            "num_epochs": 50,  # int; number of epochs
            "to_ram": False,  # bool; transfer the dataset to RAM for training
            "batch_size": 64,  # int; batch size
            "normalize": True,  # bool; if true, normalization statistics will be computed on the training set
            "temporal_subsampling_size": 0.85,  # float; this fraction of frames in each segment is randomly sampled at training time
            "parallel": False,  # bool; if true, the model will be trained on all gpus visible in the system
            "partition_method": "file",  # str; the train/test/val partitioning method
            "split_path": "esk_split.txt",  # str; path to a split file
        },
        "losses": {
            "ms_tcn": {
                "focal": True,  # bool; if True, focal loss will be used
                "gamma": 2,  # float; the gamma parameter of focal loss
                "alpha": 0.001,  # float; the weight of consistency loss
            },
        },
        "metrics": {
            "f1": {
                "average": "macro",  # ['macro', 'micro', 'none']; averaging method for classes
                "ignored_classes": None,  # set; a set of class ids to ignore in calculation
                "threshold_value": 0.5,  # float; the probability threshold for positive samples
            },
            "recall": {
                "average": "macro",  # ['macro', 'micro', 'none']; averaging method for classes
                "ignored_classes": None,  # set; a set of class ids to ignore in calculation
                "threshold_value": 0.5,  # float; the probability threshold for positive samples
            },
            "precision": {
                "average": "macro",  # ['macro', 'micro', 'none']; averaging method for classes
                "ignored_classes": None,  # set; a set of class ids to ignore in calculation
                "threshold_value": 0.5,  # float; the probability threshold for positive samples
            },
        },
        "model": {
            "num_f_maps": 128,  # int; number of maps
            "feature_dim": None,  # int; if not null, intermediate features are generated with this dimension
        },
        "features": {
            "keys": ["coords", "speed_joints", "acc_joints", "angle", "intra_distance", "coord_diff", "center", "speed_direction" ],  # set; a list of names of the features to extract
            "averaging_window": 1,  # int; if >1, features are averaged with a moving window of this size (in frames)
            "distance_pairs": None,  # list; a list of bodypart name tuples to compute distances for
            "angle_pairs": None,  # list; a list of bodypart name tuples for angle computations
            "zone_vertices": None,  # dict; zones for zone-based features
            "zone_bools": None,  # list; zone and bodypart name tuples for binary zone identifiers
            "zone_distances": None,  # list; zone and bodypart name tuples for distance computations
            "area_vertices": None,  # list; bodypart name tuples that define polygons to compute areas for
        },
    }