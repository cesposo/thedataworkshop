import time
import threading

from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.worker.node import WorkerNode
from dist_llm_train.controller.job_manager import JobManager

def run_simulation():
    """
    Runs a simulation of the distributed training system.
    """

    # Create the main controller
    controller = MainController(host='localhost', port=8000)

    # Create a few worker nodes
    worker1 = WorkerNode(name='worker-1', memory=16, flops_per_second=100, network_bandwidth=1000,
                         host='localhost', port=8001, controller_address='http://localhost:8000')
    worker2 = WorkerNode(name='worker-2', memory=32, flops_per_second=200, network_bandwidth=1000,
                         host='localhost', port=8002, controller_address='http://localhost:8000')

    # Register workers with the controller
    controller.register_worker(worker1.id, f'http://{worker1.communicator.host}:{worker1.communicator.port}', worker1)
    controller.register_worker(worker2.id, f'http://{worker2.communicator.host}:{worker2.communicator.port}', worker2)

    # Create a job manager and submit a job
    job_manager = JobManager(controller)
    job_manager.submit_job({
        'name': 'my-training-job',
        'num_tasks': 4
    })

    # Start sending heartbeats from workers
    def send_heartbeats(worker):
        while True:
            worker.send_heartbeat()
            time.sleep(5)

    heartbeat_thread1 = threading.Thread(target=send_heartbeats, args=(worker1,))
    heartbeat_thread2 = threading.Thread(target=send_heartbeats, args=(worker2,))
    heartbeat_thread1.daemon = True
    heartbeat_thread2.daemon = True
    heartbeat_thread1.start()
    heartbeat_thread2.start()

    # Run the scheduling cycle periodically
    for i in range(5):
        controller.run_scheduling_cycle()
        time.sleep(10)

    # Get the final system status
    controller.get_system_status()

    # Stop the servers
    controller.communicator.stop_server()
    worker1.communicator.stop_server()
    worker2.communicator.stop_server()

if __name__ == '__main__':
    run_simulation()
