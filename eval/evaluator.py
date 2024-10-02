import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import eval.eval_utils as utils
from utils.llm_chain import dict_to_prompt_text
import copy


class Eval:
    """
    The Eval class is responsible to calculate the score and the large errors
    """

    def __init__(self, config, analyzer=None, metric_handler=None, label_schema=None):
        """
        Initialize a new instance of the Eval class.
        :param config: The configuration file (EasyDict)
        :analyzer (optional): A chain that analyze the errors
        :metric_handler (optional): The metric handler that generate the metrics
        :label_schema (optional): The label schema
        """
        self.score_function_name = config.function_name
        self.num_errors = config.num_large_errors
        self.error_threshold = config.error_threshold
        self.num_workers = config.get("num_workers", 1)
        if metric_handler is not None:
            self.metric_handler = metric_handler
        self.dataset = None
        self.mean_score = None
        self.score_info = None
        self.label_schema = label_schema
        self.errors = None
        self.history = []
        self.analyzer = analyzer
        self.config = config
        self.score_func = self.get_eval_function()

    def get_eval_function(self):
        """
        Returns the eval function
        :param config: The eval configuration
        :return: The function implementation on a record
        """
        if self.score_function_name == 'accuracy':
            return utils.set_function_from_iterrow(lambda record: record['annotation'] == record['prediction'])
        elif self.score_function_name == 'ranking':
            return utils.set_ranking_function(self.config.function_params)
        elif self.score_function_name == 'generator':
            return utils.set_multiscore_function({metric['metric_name']: metric['metric_function']
                                                  for metric in self.metric_handler.metrics},
                                                 num_workers=self.num_workers)
        elif self.score_function_name == 't2i_vlm_score':
            return utils.get_t2i_vlm_score_func(self.config)
        elif self.score_function_name == 'dsg_score':
            return utils.dsg_score_func(self.config)
        else:
            raise NotImplementedError("Eval function not implemented")

    def eval_score(self) -> float:
        """
        Calculate the score on each row and return the mean score.
        :return: The mean score
        """
        # filter out the discarded samples
        self.dataset = self.dataset[(self.dataset['prediction'] != 'Discarded') &
                                    (self.dataset['annotation'] != 'Discarded')]
        self.dataset = self.score_func(self.dataset)
        if self.score_function_name == 'generator':
            self.score_info = {metric['metric_name']: self.dataset['score_{}'.format(metric['metric_name'])].mean()
                               for metric in self.metric_handler.metrics}
        self.mean_score = self.dataset['score'].mean()
        return self.mean_score

    def get_max_score(self, warmup=0):
        """
        Return the maximum 'mean score' (with respect to all history epochs, starting form warmup, up to last) and the epoch index of the maximum score
        :return: The epoch index of the maximum score, and the maximum score
        """
        max_idx = np.argmax([epoch['score'] for epoch in self.history[warmup:-1]])
        max_idx += warmup
        return max_idx, self.history[max_idx]['score']

    def large_error_to_str(self, error_df: pd.DataFrame, num_large_errors_per_label: int) -> str:
        """
        Return a string that contains the large errors
        :param error_df: A dataframe contains all the mislabeled samples
        :param num_large_errors_per_label: The (maximum) number of large errors per label
        :return: A string that contains the large errors that is used in the meta-prompt
        """
        if self.score_function_name == 'generator':
            error_df = error_df[:num_large_errors_per_label]
            txt_res = ''
            metrics_list = [metric['metric_name'] for metric in self.metric_handler.metrics]
            for index, row in error_df.iterrows():
                metric_result = ''
                for metric in metrics_list:
                    if row['score_{}'.format(metric)] < 5:
                        metric_result += f"#{metric}: {row['score_{}'.format(metric)]}\n{metric} score reason: {row['reasoning_{}'.format(metric)]}\n"
                txt_res += f"###Sample text\n{row['text']}\n###Agent response issues:\n{metric_result}\n"
            return txt_res
        if self.score_function_name == 't2i_vlm_score':
            error_df = error_df[:num_large_errors_per_label]
            txt_res = ''
            for index, row in error_df.iterrows():
                txt_res += f"###Sample {index} score: {row['score']}\n###Sample {index} evaluator feedback:\n{row['score_reasoning']}\n#########\n"
            return txt_res
        if self.score_function_name == 'dsg_score':
            error_df = error_df[:num_large_errors_per_label]
            text_res = ''
            for index, row in error_df.iterrows():
                text_res += f"###Sample {index} score: {row['score']}\n###Sample {index} evaluator feedback:\n{row['score_reasoning']}\n#########\n"
            return text_res
            


        required_columns = ['annotation', 'text', 'score', 'prediction']
        label_schema = error_df['annotation'].unique()
        if self.score_function_name == 'ranker':
            gt_name = 'Rank:'
        else:
            gt_name = 'GT:'
        error_res_df_list = []
        txt_res = ''
        for label in label_schema:
            cur_df = error_df[error_df['annotation'] == label]
            cur_df = cur_df.sample(frac=1.0, random_state=42)[:num_large_errors_per_label]
            error_res_df_list.append(cur_df[required_columns])
        if len(error_res_df_list) > 0:
            error_res_df = pd.concat(error_res_df_list, ignore_index=True)
            error_res_df = error_res_df.sample(frac=1.0, random_state=42)
            for i, row in error_res_df.iterrows():
                txt_res += f"Sample: {row.text}\nPrediction: {row.prediction}, {gt_name}: {row.annotation}\n#\n"
        return txt_res

    def sample_to_text(self, sample: dict, num_errors_per_label: int = 0, is_score: bool = True) -> str:
        """
        Return a string that organize the information of from the step run for the meta-prompt
        :param sample: The eval information for specific step
        :param num_errors_per_label: The max number of large errors per class that will appear in the meta-prompt
        :param is_score: If True, add the score information to the meta-prompt
        :return: A string that contains the information of the step run
        """
        prompt_str = dict_to_prompt_text(sample['prompt'], style='#')
        if is_score:
            if 'score_info' in sample.keys() and sample['score_info'] is not None:
                score_str = dict_to_prompt_text(sample['score_info'])
            else:
                score_str = f"Score: {sample['score']:.2f}\n"
            prompt_str = dict_to_prompt_text(sample['prompt'], style='#')
            return f"####\n##Prompt info:\n{prompt_str}##Prompt score:\n{score_str}#################\n"
        else:
            return f"####\n##Prompt info:\n{prompt_str}\n{self.large_error_to_str(sample['errors'], num_errors_per_label)}####"
    def add_history(self, prompt: dict, task_metadata: dict):
        """
        Add the current step information to the history
        :param prompt: The current prompt
        :param task_metadata: The task metadata
        """
        conf_matrix = None
        prompt_input = copy.deepcopy(task_metadata)
        prompt_input.update(prompt)

        large_error_to_str = self.large_error_to_str(self.errors, self.num_errors)
        prompt_input.update({
                        'accuracy': str(self.mean_score),
                        'failure_cases': large_error_to_str})
        if self.score_function_name == 'accuracy':
            conf_matrix = confusion_matrix(self.dataset['annotation'],
                                           self.dataset['prediction'],
                                           labels=self.label_schema)
            conf_text = f"Confusion matrix columns:{self.label_schema} the matrix data:"
            for i, row in enumerate(conf_matrix):
                conf_text += f"\n{self.label_schema[i]}: {row}"
            prompt_input['confusion_matrix'] = conf_text
        elif self.score_function_name == 'generator':
            prompt_input['metrics_info'] = self.metric_handler.get_metrics_info()
            prompt_input['labels'] = self.label_schema
            prompt_input['accuracy'] = dict_to_prompt_text(self.score_info)
            # TODO: Need to modify also the large_error_to_str and add there the reason for the error
        elif self.score_function_name == 'ranking':
            prompt_input['labels'] = self.label_schema

        if self.analyzer is not None:
            analysis = self.analyzer.invoke(prompt_input)
        else:
            analysis = {"text": "generated image is not good enough"}  # This is where the VLM output goes
        self.history.append({'prompt': prompt,
                             'score': self.mean_score, 'score_info': self.score_info,
                             'errors': self.errors,
                             'confusion_matrix': conf_matrix,
                             'analysis': analysis['text']})

    def extract_errors(self) -> pd.DataFrame:
        """
        Extract the errors from the dataset
        :return: records that contains the errors
        """
        df = self.dataset
        err_df = df[df['score'] < self.error_threshold]
        err_df = err_df.sort_values(by='score', ascending=True)
        self.errors = err_df
        return self.errors

    def extract_correct(self) -> pd.DataFrame:
        """
        Extract the correct samples from the dataset
        :return: records that contains the correct samples
        """
        df = self.dataset
        return df[df['score'] > self.error_threshold]

    def extract_boundary_predictions(self) -> pd.DataFrame:
        """
        Extract boundary samples on which the model is uncertain
        :return: records that contains boundary samples
        """
        pass
