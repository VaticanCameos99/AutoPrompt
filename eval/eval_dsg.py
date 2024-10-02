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

        #Get source image questions
        self.get_source_image_questions()

    def get_source_image_questions(self):

        for i, filename in enumerate(os.listdir(self.source_images)):
            file_path = os.path.join(self.source_images, filename)
            message_content = [{"type": "text", "text": "You are given images of a specific character or an object, we refer these images as the 'source images' and the object as <x>. Identify elements that are in the foreground and then generate yes/no type questions to check if these elements are in the image.\
                                If there is a human in the foreground, you must ask questions like: Is there a woman in the image? or Is there a man in the image? - Depending on which of the two is found in the image \
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
            "text": """Given the following image, answer the question given in either 1 (for yes) or 0 (for no) only. Once you are done generating the answers, compile the observations and generate a feedback of the results. Feedback should suggest what should be there since it is missing:\
            Example :
            Consider the questions: Is there a racoon? , Is the racoon wearing a hat? , Is the racoon sitting on a table?
            Output format expected:
            #Answers:
            1. 1
            2. 0
            3. 1
            #Feedback: The racoon should be wearing a hat which is missing.
              Questions: {}""".format(str(questions))},
            {"type": "image_url", "image_url": {"url": url}}]
        return ChatPromptTemplate.from_messages([HumanMessage(content=message_content)])
        
    def generate_dsg(self, prompts):
        #Save with index
        inputs = prompts.set_index('id')['text'].apply(lambda x: {'input': x}).to_dict()
        #calls to llm are concurrent
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

    def dataset_invoke(self, datasets):
        #compare image, question and dependency for globals
        prompts = datasets[['id', 'text']]

        def has_question_empty_or_missing(row):
            my_dict_entry = self.results.get(row['id'], {})
            return 'question' not in my_dict_entry
        
        filtered_prompts = prompts[prompts.apply(has_question_empty_or_missing, axis=1)]
        if (not filtered_prompts.empty):
            self.generate_dsg(filtered_prompts)

        self.create_qna_prompts(datasets)
        datasets['score'] = datasets['id'].map(self.results.get).apply(lambda x: x['score'] if x else None)
        datasets['score_reasoning'] = datasets['id'].map(self.results.get).apply(lambda x: x['score_reasoning'] if x else None)
        return datasets
        
    def get_score(self, datasets):
        for key in self.results.keys():
            answer = self.results[key]['answers']
            vaidity_count = self.results[key]['vaidity_count']

            self.results[key]['score'] = sum(answer) / vaidity_count
        return           
    
    def create_qna_prompts(self, dataset: pd.DataFrame):
         
         #TODO:
         def extract_answers_and_feedback(text):
            feedback_index = text.find('#Feedback:')
            answers_index = text.find('#Answers:')

            answers = text[answers_index+len('#Answers:') : feedback_index]
            answers_list = [int(line.split('. ')[1]) for line in answers.split('\n') if line.strip() != '']

            feedback = text[feedback_index+len("#Feedback:") : ]

            return answers_list, feedback

         answers = {}
         batch_inputs = []
         batch_inputs_src_questions = []
         for key in self.results.keys():
            src_questions = self.results[key]['source_questions'] #Format: ["Q1", "Q2", ...]
            questions = self.results[key]['question'] #Format: 1: "Q1", 2: "Q2" .. 
            row = dataset[dataset['id'] == key]
            batch_inputs.append({'sample_chain_input': '', 'index': key,
                                 'chain': DSGWrapper(self.generate_qna_prompt(questions, row['prediction'].values[0]) | self.llm)})
            batch_inputs_src_questions.append({'sample_chain_input': '', 'index': key,
                                 'chain': DSGWrapper(self.generate_qna_prompt(src_questions, row['prediction'].values[0]) | self.llm)})

         answers = sync_chain_batch_run(None, batch_inputs, self.config.num_workers, get_index=True)
         src_answers = sync_chain_batch_run(None, batch_inputs_src_questions, self.config.num_workers, get_index=True)

         for d in answers:
            index = d['index']
            result_string = d['result']
            answers_list, feedback = extract_answers_and_feedback(result_string)
            self.results[index]['answers'] = answers_list
            self.results[index]['score_reasoning'] = feedback

         #Disqualify invaid questions and answers
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
                    answers[id] = 0
                    total_possible_score -=1

            self.results[key]['answers'] = answers
            assert len(self.results[key]['answers']) == len(self.results[key]['question']), str(len(self.results[key]['answers'])) + " & " + str(len(self.results[key]['question']))
            self.results[key]['score'] = sum(answers)
            self.results[key]['total_possible_score'] = total_possible_score

        #Processing answers to source questions
         for d in src_answers:
            index = d['index']
            result_string = d['result']
            answers_list, feedback = extract_answers_and_feedback(result_string)
            self.results[key]['score'] += sum(answers_list)
            self.results[key]['total_possible_score'] += len(answers_list)

            scaled_score = (self.results[key]['score'] / self.results[key]['total_possible_score']) * 10
            self.results[key]['score'] = scaled_score
            self.results[index]['score_reasoning'] += feedback
            del self.results[key]['total_possible_score']
        
         return