import gc
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import filterfalse


class Output:
    has_epsilon = False

    def __init__(self, file_input: str, mode: str = 'results', output_every_n=100, random_overlay=False):
        """ mode: either 'results' or 'model'.
            ... -o '<path>/<file>.txt'  # outputs experiment data to txt format
            ... -o '<path>/<file>.csv'  # outputs experiment data to csv format
            ... -o '<path>/<file>.png'  # outputs experiment data chart (provided matplotlib) to png or jpg format
            ... -o '<path>/<file>.h5'   # when mode == 'model', outputs neural network model here.
            ... -o '<path>'             # when there is no file extension (.xx),
                                        # a .txt file, a .csv file and (if mode == 'model')
                                            a list of .h5 file checkpoints are stored in this folder.
            ... -o None
             or -o 'None'               # show plotted chart on screen

            # by default, there will be printing onto the console. there is no way to turn it off.
            """
        self.mode = mode
        Output.output = (file_input is not None)
        if not self.output:
            return

        file = file_input.rsplit(".", 1)
        file[0] = "./" + file[0]
        self.files = {}
        self.output_every_n = output_every_n
        self.random_overlay = random_overlay
        self.df = pd.DataFrame({'episode': [],
                                'step': [],
                                'score': [], 'epsilon': [], 'good': [], 'best': []})
        if file_input.endswith(".txt"):
            self.files.update({"txt": file[0].rsplit('/', 1)[1]})
        elif file_input.endswith(".csv"):
            self.files.update({"csv": file[0].rsplit('/', 1)[1]})
        elif file_input.endswith(".png"):
            self.files.update({"png": file[0].rsplit('/', 1)[1]})
        elif file_input.endswith(".jpg"):
            self.files.update({"jpg": file[0].rsplit('/', 1)[1]})
        elif file_input.endswith(".h5") and mode == 'model':
            self.save_to_single = True
            self.files.update({"h5": file[0].rsplit('/', 1)[1]})
        elif file_input.endswith(".h5s") and mode == 'model':
            self.save_to_single = False
            self.files.update({"h5": file[0].rsplit('/', 1)[1]})
        else:
            file[0] = file[0][2:] if file[0].startswith("./") else file[0]
            if "." not in file_input: # is a directory
                self.files.update({"txt": "data", "csv": "data", "png": "evaluation_graph"})
                if mode == 'model':
                    self.files.update({"h5": "model"})
                file = [file_input+"/"]
            else:
                print(f"File format {file[1]} not supported. "
                      f"Train data and files (if any) is now saved to the directory {file[0]}.")
            try:
                os.mkdir(file[0])
            except Exception as e:
                print(e)
        file[0] = file[0][2:] if file[0].startswith("./") else file[0]
        file_input = file[0].rsplit("/", 1) if "/" in file[0] else (".", file[0])
        # ignore file format if the format is not supported
        self.dir = file_input[0]
        print(f"The following items will be outputted in folder {self.dir}:\n"
              + '\n'.join(map(lambda x: '\t- '+x[1]+'.'+x[0], self.files.items())))

        self.actions = {"txt": self.output_txt, "csv": self.output_csv, "png": self.output_img,
                        "jpg": self.output_img, "h5": self.output_model}
        self.actions = dict(list(filterfalse(lambda x: x[0] not in self.files, self.actions.items())))

        matplotlib.use('Agg')

    @staticmethod
    def check(func):
        def inner(*args, **kwargs):
            if not Output.output:
                return
            return func(*args, **kwargs)
        return inner

    @check
    def output_txt(self, output="", **kwargs):
        with open(f"{self.dir}/{self.files['txt']}.txt", "a+") as f:
            f.write(output + "\n")

    @check
    def output_csv(self, episode=0, step=0, info=None, **kwargs):
        self.df.to_csv(f"{self.dir}/{self.files['csv']}.csv")

    @check
    def output_img(self, episode=0, output_every_n=1, do_output=True, **kwargs):
        if self.mode != 'model' and episode >= 0 or not do_output:
            return
        """
        # there is some issue with plt and pygame.
        # plt resizes the pygame window by ignoring Window's screen resize ratio (200% in my display)
        # and thus shrinks the window size of pygame after every plt call.
        # Currently, I cannot find any resources online that could help me solve this problem, so
        # this would NOT be implemented directly. Only the last data.png is saved.
        """
        if episode % output_every_n != output_every_n-1 and episode >= 0:
            return
        mode = 'png' if 'png' in self.files else 'jpg'

        fig, axs = plt.subplots(nrows=3 if self.has_epsilon else 2, ncols=2,
                                figsize=(8, 9 if self.has_epsilon else 6), num=1, clear=True)
        fig.tight_layout()
        self.df = self.df.sort_values(by='episode')
        self.df.set_index('episode')
        rolling = self.df.rolling(10, on='episode').mean()

        # 'best' is used as 'random_score' when self.random_overlay is True.
        axs[0, 0].plot(rolling['score'].dropna())
        try:
            if self.random_overlay:
                axs[0, 0].plot(rolling['best'], alpha=.5)
                axs[0, 0].legend(['score', 'random'])
            else:
                axs[0, 0].plot(self.df['good'].dropna())
                axs[0, 0].plot(self.df['best'].dropna())
                axs[0, 0].legend(['score', 'good', 'best'])
        except KeyError:
            pass
        self.df['score'].hist(ax=axs[0, 1], range=[0, 5000], bins=100)
        if self.random_overlay:
            self.df['best'].hist(ax=axs[0, 1], range=[0, 5000], bins=100, alpha=0.5)
        axs[0, 1].set_xlabel("score")
        axs[0, 1].set_ylabel("frequency")

        # 'good' is used as 'random_step' when self.random_overlay is True.
        axs[1, 0].plot(rolling['step'].dropna())
        if self.random_overlay:
            axs[1, 0].plot(rolling['good'], alpha=.5)
            axs[1, 0].legend(['step', 'random'])
        self.df['step'].hist(ax=axs[1, 1], range=[-5, 2000], bins=100)
        if self.random_overlay:
            self.df['good'].hist(ax=axs[1, 1], range=[-5, 2000], bins=100, alpha=0.5)
        axs[1, 1].set_xlabel("step")
        axs[1, 1].set_ylabel("frequency")
        if self.has_epsilon:
            axs[2, 0].set_ylim([0, 1])
            self.df['epsilon'].plot(ax=axs[2, 0])
            self.df['epsilon'].hist(ax=axs[2, 1], range=[0, 1], bins=100)
            axs[2, 1].set_xlabel("epsilon")
            axs[2, 1].set_ylabel("frequency")
        del rolling

        plt.savefig(f"{self.dir}/{self.files[mode]}.{mode}")
        plt.close(fig)
        plt.close("all")
        gc.collect()

    @check
    def output_model(self, episode=0, model=None, output_every_n=1, **kwargs):
        if episode % output_every_n != output_every_n-1 and episode >= 0:
            return
        if episode == -1:
            filename = f"{self.dir}/{self.files['h5']}_best.h5"
        elif episode < 0:
            filename = f"{self.dir}/{self.files['h5']}_best_replacedAt{-episode}.h5"
        else:
            filename = f"{self.dir}/{self.files['h5']}_epoch{episode}.h5"
        model.save(filename)

    def log(self, done, episode, step, info, model=None, do_output=True, epsilon: float = -1., training=False,
            best: int = None):
        if done:
            output = f"Episode {episode} succeeded in {step} steps with score {info['score']}... epsilon {epsilon} ... training {training}"
        else:
            output = f"Episode {episode} truncated ... in {step} steps with score {info['score']} ... epsilon {epsilon}"

        print(output)

        if not self.output:
            return
        self.has_epsilon = epsilon != -1
        self.concat({'episode': [episode], 'step': [step], 'score': [info['score']],
                     'epsilon': [epsilon], 'best': [np.NAN], 'good': [np.NAN]})
        [a(output=output, episode=episode, step=step, info=info, output_every_n=self.output_every_n,
           model=model, do_output=do_output) for a in self.actions.values()]

    @check
    def logs(self, output: str):
        if "txt" in self.actions:
            self.output_txt(output)

    def concat(self, dic: dict):
        self.df = pd.concat(
            [self.df, pd.DataFrame(dic)],
            ignore_index=True
        ).groupby('episode').sum(numeric_only=True, min_count=1).reset_index()





