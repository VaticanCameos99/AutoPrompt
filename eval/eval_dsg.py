from dsg.query_utils import generate_dsg # type: ignore
from dsg.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output #type: ignore
# from dsg.vqa_utils import MPLUG # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from eval.eval_im import ImChainWrapper
from utils.llm_chain import sync_chain_batch_run
import pandas as pd
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_llm, override_config
import base64

class DSGWrapper:
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, kwargs):
        result = self.chain.invoke({})
        return result.content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class DSG:
    def __init__(self, config):
        self.llm = get_llm(config.llm)
        self.prompt = config.task_description
        self.config = config
        self.source_images = config.source_image_folder
        self.results = {}
        # self.vqa_model = MPLUG()
        self.get_source_image_questions()

    def get_source_image_questions(self):
        # message_content = [{"type": "text", "text": "You are given images of a specific character or an object, we refer these images as the 'source images' and the object as <x>"}]
        for i, filename in enumerate(os.listdir(self.source_images)):
            file_path = os.path.join(self.source_images, filename)
            message_content = [{"type": "text", "text": "You are given images of a specific character or an object, we refer these images as the 'source images' and the object as <x>. Identify elements that are in the foreground and then generate yes/no type questions to check if these elements are in the image.\
                                Eg: Suppose you identify that an image has a racoon wearing a hat and that the racoon is sitting on a table. The questions generated you generate as output would be: \
                                1. Is there a racoon in the image? \
                                2. Is the racoon wearing a hat? \
                                3. Is the racoon sitting on a table? \
                                4. Is the color of the racoon grey? \
                                and so on ..."}]
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                try:
                    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"
                                                                               f"{encode_image(file_path)}"}})
                    prompts = [HumanMessage(content = message_content)]
                    description = self.llm.invoke(prompts).content
                    src_questions = [s.strip().split('. ', 1)[1] for s in description.split('\n')]
                    self.results[i] = {'source_questions':  src_questions}
                except Exception as e:
                        print(f"Failed to process {filename}: {e}")
        return

    #test performance for google
    def generate_qna_prompt(self, questions, url):
        message_content = [{
            "type": "text",
            "text": """Given the following image, answer the question given in either 1 (for yes) or 0 (for no) only.\
            Example Output Format:
            1. 1
            2. 0
            3. 1
              Questions: {}""".format(str(questions))},
            {"type": "image_url", "image_url": {"url": url}}]
        #Temporary, will change to system and human message later
        return ChatPromptTemplate.from_messages([HumanMessage(content=message_content)])
        
        #where is the image url?

    def generate_dsg(self, prompts):
        # selresults = {}
        #Save with index
        inputs = prompts.set_index('id')['text'].apply(lambda x: {'input': x}).to_dict()

        #calls to llm are concurrent
        time.sleep(4)
        id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
            inputs,
            generate_fn=self.llm.invoke, 
            verbose = False
            )
        
        #Parser is not concurrent
        for key in inputs.keys():
            self.results[key]['tuple'] = parse_tuple_output(id2tuple_outputs[key]['output'])
            self.results[key]['question'] = parse_question_output(id2question_outputs[key]['output'])
            self.results[key]['dependency'] = parse_dependency_output(id2dependency_outputs[key]['output'])

        # save as self.results
        #Shouldn't results be created at run time? "isn't prompt in config?"
        # self.results = results
        # create prompts based on these results
        # prompt is created when you invoke the chain, so add it when you call

        # create call to vqa, that adds score to dataset
        # return results

    def dataset_invoke(self, datasets):
        #compare image, question and dependency for globals

        #Datasets structure if there are multiple images. Do we only save the last state? 
        # datasets_copy = datasets.copy()
        # print('datasets:\n', datasets.columns)
        prompts = datasets[['id', 'text']]
        self.generate_dsg(prompts)
        # print('results', self.results)
        self.create_qna_prompts(datasets)
        # self.get_score(datasets) 
        datasets['score'] = datasets['id'].map(self.results.get).apply(lambda x: x['score'] if x else None)
        # self.results = {}
        print(self.results, 'self.results')
        print(datasets, 'datasets')
        return datasets
        
    def get_score(self, datasets):
        # output_list = []
        for key in self.results.keys():
            answer = self.results[key]['answers']
            vaidity_count = self.results[key]['vaidity_count']

            self.results[key]['score'] = sum(answer) / vaidity_count
            # output_list.append({'key': key, 'score':  sum(answer) / vaidity_count})
        return           
    
    def create_qna_prompts(self, dataset: pd.DataFrame):
         answers = {}
         batch_inputs = []
         batch_inputs_src_questions = []
         for key in self.results.keys():
            src_questions = self.results[key]['source_questions']
            questions = self.results[key]['question'] #Format: 1: "Q1", 2: "Q2" .. 
            # questions.extend(src_questions) #Best to separate this and append to answers later
            row = dataset[dataset['id'] == key]
            batch_inputs.append({'sample_chain_input': '', 'index': key,
                                 'chain': DSGWrapper(self.generate_qna_prompt(questions, row['prediction'].values[0]) | self.llm)})
            batch_inputs_src_questions.append({'sample_chain_input': '', 'index': key,
                                 'chain': DSGWrapper(self.generate_qna_prompt(src_questions, row['prediction'].values[0]) | self.llm)})
            

                #  answer = self.vqa_model.vqa(generated_image, question)

         answers = sync_chain_batch_run(None, batch_inputs, self.config.num_workers, get_index=True)
         time.sleep(4)
         src_answers = sync_chain_batch_run(None, batch_inputs, self.config.num_workers, get_index=True)
         print('all res', answers)
        #  parse result, put in index-array format
         for d in answers:
            index = d['index']
            result_string = d['result']
            result_list = [int(line.split('. ')[1]) for line in result_string.split('\n') if line]
            self.results[index]['answers'] = result_list

         #Disqualify invaid questions and answers
         #TODO: after correcting validity, consider parallelising this
         #Total possible points: valid questions + src questions
         #total score = sum(answers) = sum(src_answers)
         #next, scale total score to out of 100
         for key in self.results.keys():
            dependencies = self.results[key]['dependency']
            answers = self.results[key]['answers']
            total_possible_score = len(answers)
            for id, parent_ids in dependencies.items():
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == 0:
                        continue
                    if answers[parent_id] == 0:
                        any_parent_answered_no = True
                        break
                if any_parent_answered_no:
                    print(answers, 'answers')
                    print(id, 'id')
                    answers[id] = 0
                    #TODO: consider nulling out answers instead of saving validity
                    total_possible_score -=1

            self.results[key]['answers'] = answers
            assert len(self.results[key]['answers']) == len(self.results[key]['question']), str(len(self.results[key]['answers'])) + " & " + str(len(self.results[key]['question']))
            # print('sum validty', sum(validity))
            # print('len source questions', len(self.results[key]['source_questions']))
            self.results[key]['score'] = sum(answers)
            self.results[key]['total_possible_score'] = total_possible_score
            # print(self.results, 'self results after validity')
            # paes
         for d in src_answers:
            index = d['index']
            result_string = d['result']
            result_list = [int(line.split('. ')[1]) for line in result_string.split('\n') if line]
            # self.results[index]['answers'].extend(result_list)
            self.results[key]['score'] += sum(result_list)
            #todo: add scaling for each id and save as score
            self.results[key]['total_possible_score'] += len(result_list)

            scaled_error = 10 - (self.results[key]['score'] / self.results[key]['total_possible_score']) * 10
            self.results[key]['score'] = scaled_error
            del self.results[key]['total_possible_score']


         print(self.results, 'self results')
         return


        
if __name__ == "__main__":
    config_params = override_config("/Users/nirvivakharia/Documents/CVPRPaper2024/AutoPrompt/config/config_diff/config_images.yml")
    # print(config_params.eval)
    config_params.eval['task_description'] = """A raccoon wearing formal clothes, wearing a tophat
and holding a cane. The raccoon is holding a garbage
bag. Oil painting in the style of Vincent Van Gogh""".strip()
    dsg = DSG(config_params.eval)
    print('gemini', dsg.llm)
    # print(dsg.llm.invoke("Introduce yourself").content)
    print('dsg op', dsg.generate_dsg([config_params.eval['task_description']], verbose = False))

    #promptsi+1 in dataset
    #