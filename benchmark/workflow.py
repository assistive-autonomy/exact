from dlc2action.project import Project
from loguru import logger

PROJECT_NAME = "esk"  # the name of the project
PROJECTS_PATH = "dlc2a_benchmark"  # the path to the projects folder
DATA_TYPE = "dlc_track"  # choose from Project.print_data_types()
ANNOTATION_TYPE = "dlc"  # choose from Project.print_annotation_types()
DATA_PATH = "/home/rimvydas/exact/data/esk_30/D2A_converted_pose_smpl"  # path to data files
ANNOTATION_PATH = "/home/rimvydas/exact/data/esk_30/D2A_converted_label_verbs"  # path to annotation files

Project.remove_project("esk", projects_path=PROJECTS_PATH)

logger.info(f"Initializing project '{PROJECT_NAME}' with data type '{DATA_TYPE}' and annotation type '{ANNOTATION_TYPE}'")

project = Project(
    PROJECT_NAME,
    projects_path=PROJECTS_PATH,
    data_type=DATA_TYPE,
    annotation_type=ANNOTATION_TYPE,
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
)

logger.info("updating project parameters...")

project.update_parameters(
    {
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
            "split_path": "/home/rimvydas/exact/benchmark/traintest_split.txt",  # str; path to a split file
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
    },
)

MODELS = [
    "transformer",
]  

logger.info("Starting hyperparameter search, training, and evaluation...")
for model in MODELS:  # run a hyperparameter search for your models
    project.run_default_hyperparameter_search(
        f"{model}_search",
        model_name=model,
        num_epochs=3,
        n_trials=10,
        metric="f1",
    )

for model in MODELS:  # train models with the best hyperparameters
    project.run_episode(
        f"{model}_best",
        load_search=f"{model}_search",
        parameters_update={"general": {"model_name": model}},
        n_seeds=3,
        force=True,
    )

project.plot_episodes(  # compare training curves
    [f"{model}_best" for model in MODELS],
    metrics=["f1"],
    save_path="training_curves.png",
    title="Best model training curves",
)

for model in MODELS:  # evaluate more metrics
    project.evaluate(
        [f"{model}_best"],
        parameters_update={
            "general": {"metric_functions": ["segmental_f1", "pr-auc", "f1"]},
            "metrics": {"f1": {"average": "none"}},
        },
    )

results_df = project.get_results_table(
    [f"{model}_best" for model in MODELS]
)  # get a table of the results

results_df.to_csv("results_table.csv", index=False)


