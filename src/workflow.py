from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Exalearn_miniapp_EnTK_parallel_CPU_GPU")

        parser.add_argument('--n_ensemble_per_task', type=int, default=5,
                            help='the number of lora ensemble to train per task (default: 5)')
        parser.add_argument('--model_name', default='Llama3',
                            help='the name of base LLM (default: llama3)')
        parser.add_argument('--seed_base', type=int, default=237,
                            help='the base seed, each task will use a different seed (default: 237)')
        parser.add_argument('--n_train_task', type=int, default=1,
                            help='the number of training tasks launched in parallel (default: 1)')
        parser.add_argument('--tmp_dir', required=True,
                            help='the dir where train task save metric output')
        parser.add_argument('--work_dir', required=True,
                            help='working dir where the source code is located')

        args = parser.parse_args()
        self.args = args

    # This is for simulation, return a sim task
    def run_train(self, seed):

        t = entk.Task({'uid': ru.generate_id("train")})
        t.pre_exec = [
                "source /global/homes/t/tianle/useful_script/conda_surp_2024",
                ]
        t.executable = 'python3'
        t.arguments = ['{}/run_model.py'.format(self.args.work_dir),
                       '--model_name={}'.format(self.args.model_name),
                       '--n_ensemble={}'.format(self.args.n_ensemble_per_task),
                       '--tmp_dir={}'.format(self.args.tmp_dir),
                       '--seed={}'.format(seed)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 4,
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
                }

        return t


    def run_process(self):

        t = entk.Task({'uid': ru.generate_id("process")})
        t.pre_exec = [
                "source /global/homes/t/tianle/useful_script/conda_surp_2024",
                ]
        t.executable = 'python3'
        t.arguments = ['{}/process.py'.format(self.args.work_dir),
                       '--tmp_dir={}'.format(self.args.tmp_dir)]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 4,
            'cpu_thread_type'   : rp.OpenMP
                }

        return t

    def generate_pipeline(self):

        p = entk.Pipeline()

        s0 = entk.Stage()
        for i in range(self.args.n_train_task):
            t0 = self.run_train(self.args.basic_seed + i)
            s0.add_tasks(t0)
        p.add_stages(s0)

        s1 = entk.Stage()
        t1 = self.run_process()
        s1.add_tasks(t1)
        p.add_stages(s1)

        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":
    
    mvp = MVP()
    mvp.set_resource(res_desc = {
        'resource': 'nersc.perlmutter_gpu',
        'queue'   : 'premium',
        'walltime': 120, #MIN
        'cpus'    : int(4 * (mvp.args.n_train_task + 1)),
        'gpus'    : mvp.args.n_train_task,
        'project' : m2616_g
        })
    mvp.run_workflow()

