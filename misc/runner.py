import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import toml
from joblib import Parallel, delayed
from tqdm import tqdm

from .FuncAnimationWithEndFunc import FuncAnimationWithEndFunc
from . import decorate_print, make_parent_dir

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, model_class, params, change_cwd=True):
        if isinstance(params, Path) or isinstance(params, str):
            params_filename = params
            self.params = toml.load(params_filename)
            self.params["general"]["filename"] = str(Path(params_filename).resolve())
        elif isinstance(params, dict):
            self.params = params
        else:
            raise ValueError(
                f"params must be pathlib.Path, str or dict: {type(params)}"
            )

        pgen = self.params["general"]
        date_now = datetime.now()
        # logger.info(
        #     f"{pgen['description']} | @ {date_now.strftime('%Y/%m/%d %H:%M:%S')}"
        # )
        # logger.debug(f"Args: {sys.argv}")

        self.max_timestep = pgen["step_max"]
        self.original_cwd = os.getcwd()
        if change_cwd:
            dir_name = Path(
                self.original_cwd + "/outputs/" + date_now.strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(dir_name)
            os.chdir(dir_name)

        self.model = model_class(self.params)

    def run(self, callback) -> None:
        start = time.monotonic()
        self.model.draw_initial()
        with tqdm(total=self.max_timestep) as pbar:
            fanm = FuncAnimationWithEndFunc(
                self.model.fig,
                self.update_animation,
                fargs=(pbar,),
                init_func=(lambda: None),  # visualize results when t=1
                interval=self.interval,
                frames=self.max_timestep,
                end_func=plt.close,
            )
            callback(fanm)
        self.elapsed_time = time.monotonic() - start

    def display(self):
        self.interval = self.params["display"]["interval"]
        self.run(lambda fanm: plt.show())

    def run_silent(self, plt_close=True):
        start = time.monotonic()
        self.model.draw_initial()
        for _ in range(self.params["general"]["step_max"]):
            self.model.step()
            if not self.model.running:
                break
        if plt_close:
            plt.close()
        self.elapsed_time = time.monotonic() - start

    def run_silent_wo_plt(self):
        start = time.monotonic()
        for _ in range(self.params["general"]["step_max"]):
            self.model.step()
            if not self.model.running:
                break
        self.elapsed_time = time.monotonic() - start

    def run_save_picture(self, dst_dir, plt_close=True):
        start = time.monotonic()
        self.model.draw_initial()
        dst = Path(dst_dir)
        if not dst.exists():
            dst.mkdir()

        for i in range(self.params["general"]["step_max"]):
            self.model.step()
            self.model.draw()
            plt.savefig(dst / f"{i:03d}.png")
            if not self.model.running:
                break
        if plt_close:
            plt.close()
        self.elapsed_time = time.monotonic() - start

    def update_animation(self, step, pbar):
        self.model.step()
        self.model.draw_successive()
        pbar.update(1)

    def make_movie(self, filename, writer="ffmpeg"):
        pmovie = self.params["movie"]
        self.interval = pmovie["interval"]
        make_parent_dir(filename)
        self.run(lambda fanm: fanm.save(filename, writer, dpi=pmovie["dpi"]))

    def get_result(self):
        return self.model.get_result()

    def get_result_description(self):
        return self.model.get_result_description()

    def log_result(self):
        with decorate_print(logger.info, "Results"):
            self.log_elapsed_time()
            self.model.log_result()

    def log_parameters(self):
        with decorate_print(logger.debug, "Parameters"):
            for line in toml.dumps(self.params).splitlines():
                logger.debug(line)

    def log_elapsed_time(self):
        hours, rem = divmod(self.elapsed_time, 3600.0)
        minutes, seconds = divmod(rem, 60.0)
        logger.info(
            f"Elapsed time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"
        )


class BatchRunner:
    def __init__(self, model_class, params, batch_params):
        if isinstance(params, Path) or isinstance(params, str):
            params_filename = params
            self.params_orig = toml.load(params_filename)
            self.params_orig["general"]["filename"] = str(
                Path(params_filename).resolve()
            )
        elif isinstance(params, dict):
            self.params_orig = params
        else:
            raise ValueError(
                f"params must be pathlib.Path, str or dict: {type(params)}"
            )

        desc = self.params_orig["general"]["description"]
        logger.info(f"{desc} | Batch @ {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        logger.debug(f"Args: {sys.argv}")

        self.model_class = model_class
        self.runner_lst = []
        self.model_class = model_class
        if isinstance(batch_params, Path) or isinstance(batch_params, str):
            batch_params_filename = batch_params
            self.batch_params = toml.load(batch_params_filename)
            self.batch_params["filename"] = str(Path(batch_params_filename).resolve())
        elif isinstance(batch_params, dict):
            self.batch_params = batch_params
        else:
            raise ValueError(
                f"batch_params must be pathlib.Path, str or dict {type(batch_params)}"
            )

        self.task_num = self.batch_params["task_num"]
        self.n_jobs = self.batch_params["n_jobs"]
        self.verbose = self.batch_params["verbose"]

        self.result_lst = None

    def log_elapsed_time(self):
        hours, rem = divmod(self.elapsed_time, 3600.0)
        minutes, seconds = divmod(rem, 60.0)
        logger.info(
            f"Elapsed time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"
        )

    @staticmethod
    def log_parameters(params, params_name):
        with decorate_print(logger.debug, f"{params_name} Parameters"):
            for line in toml.dumps(params).splitlines():
                logger.debug(line)

    def log_all_parameters(self):
        BatchRunner.log_parameters(self.batch_params, "Batch")
        BatchRunner.log_parameters(self.params_orig, "Model")

    def run_serial(self, param_generator):
        start = time.monotonic()
        self.result_lst = []
        for param in tqdm(param_generator(self.params_orig), total=self.task_num):
            self.result_lst.append(self._run(param))
        self.elapsed_time = time.monotonic() - start
        return self.result_lst

    def run_parallel(self, param_generator):
        start = time.monotonic()
        processed = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            [delayed(self._run)(param) for param in param_generator(self.params_orig)]
        )
        self.elapsed_time = time.monotonic() - start
        return processed

    def _run(self, params):
        runner = ModelRunner(self.model_class, params=params, change_cwd=False)
        runner.run_silent()
        return runner.get_result()

    def get_result_description(self):
        return self.model_class.get_result_description()
