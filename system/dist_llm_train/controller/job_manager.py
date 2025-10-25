from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.task.training_task import TrainingTask

class JobManager:
    """
    Manages the submission of training jobs to the main controller.
    """

    def __init__(self, controller: MainController):
        self.controller = controller

    def submit_job(self, job_config: dict):
        """
        Submits a new training job to the system.

        This is currently a placeholder that creates a few dummy tasks.
        """
        print(f"[JobManager] Submitting job: {job_config.get('name')}")

        # In a real implementation, this would parse the job config
        # and create a series of tasks.
        for i in range(job_config.get('num_tasks', 5)):
            task = TrainingTask(
                task_id=f"task-{i}",
                model_name=job_config.get('model_name', 'dummy-model'),
                model_layer=i,
                model_shard_size=1.0,
                data_size=0.5,
                required_flops=1000
            )
            self.controller.add_task(task)
