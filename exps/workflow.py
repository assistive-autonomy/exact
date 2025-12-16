from dlc2action.project import Project
from loguru import logger

from params import params

PROJECT_NAME = "esk"  # the name of the project
PROJECTS_PATH = "dlc2a_benchmark"  # the path to the projects folder
DATA_TYPE = "dlc_track"  # choose from Project.print_data_types()
ANNOTATION_TYPE = "dlc"  # choose from Project.print_annotation_types()
DATA_PATH = "/pvc/esk/D2A_converted_pose_smpl"  # path to data files
ANNOTATION_PATH = "/pvc/esk/D2A_converted_label_verbs"  # path to annotation files

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

project.update_parameters(params)

MODELS = [
    "ms_tcn",
    "c2f_tcn",
    "c2f_transformer",
    "edtcn",
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


