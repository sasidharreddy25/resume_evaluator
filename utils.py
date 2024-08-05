import os
import json
import time
import PyPDF2
from agents import Job_Description_Agent, Conversation_Agent, Fitment


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    return text

def collect_email_data(resume_id, job_description_path, resume_folder_path):
    job_description_text=extract_text_from_pdf(job_description_path)
    resume_list=os.listdir(resume_folder_path)
    for i in resume_list:
        if i==resume_id:
            path=resume_folder_path+"/"+resume_id
            break
    resume_text=extract_text_from_pdf(path)
    return job_description_text, resume_text

def calculate_score(scores):
    score=0
    count=0
    for i in scores:
        for j, k in i.items():
            score+=k
        count+=len(i)
    total_score=score/count
    return total_score


class Resume_Evaluator:
    def __init__(self, openai_type):
        self.job_description_agent = Job_Description_Agent(openai_type)
        self.conversation_agent = Conversation_Agent(openai_type)
        self.fitment_agent=Fitment(openai_type)
        self.job_description_path=""
        self.resume_folder_path=""
        self.resume_analysis={}
        self.score_values={}

    def _fitment_points(self, QnA):
        strength=""
        gaps=""
        questions=""
        for i in QnA:
            if i['score']==0:
                gaps+=i['recruiter']+'\n'
            elif i['score']==1:
                strength+=i['recruiter']+'\n'
            else:
                questions+=i['recruiter']+'\n'
        return strength, gaps, questions

    def evaluate(self, job_description_path, resume_folder_path):
        self.job_description_path=job_description_path
        self.resume_folder_path=resume_folder_path
        job_description_text=extract_text_from_pdf(job_description_path)
        requirements = self.job_description_agent.agent(job_description_text)
        job_requirement=[]
        for i, j in requirements.items():
            job_requirement.extend(j)
        print(f'Length of requirements: {len(job_requirement)}')
        files = os.listdir(resume_folder_path)
        for file in files:
            resume_text=extract_text_from_pdf(resume_folder_path+"/"+file)
            QnA, scores = self.conversation_agent.start_conversation(job_requirement, resume_text)
            total_score=scores/len(job_requirement)
            strength, gaps, questions = self._fitment_points(QnA)
            st_strength=self.fitment_agent.strengths(strength)
            st_gaps=self.fitment_agent.gaps(gaps)
            st_questions=self.fitment_agent.questions(questions)
            self.resume_analysis[file]={'total_score':total_score, 'strength':st_strength, 'gaps':st_gaps, 'questions':st_questions}
            self.score_values[file]=total_score
            # Sort the dictionary by values
            sorted_scores = dict(sorted(self.score_values.items(), key=lambda item: item[1], reverse=True))

            candidate_fitment=""
            for i, j in sorted_scores.items():
                candidate_dict=self.resume_analysis[i]
                candidate_fitment+=f"Resume ID:\n{i}" +'\n\n'
                candidate_fitment+=f"Resume score:\n{candidate_dict['total_score']}" +'\n\n'
                candidate_fitment+=f"Strengths:\n{candidate_dict['strength']}" +'\n'
                candidate_fitment+=f"Gaps:\n{candidate_dict['gaps']}" +'\n'
                candidate_fitment+=f"Questions_to_Candidate:\n{candidate_dict['questions']}" +'---------------------------------------------------------------------\n\n'
            print(f'Done for the {file}')
        return candidate_fitment