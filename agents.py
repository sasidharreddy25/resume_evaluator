import json
import time
import PyPDF2
from openai import OpenAI
from openai import AzureOpenAI
from config import api_key, azure_endpoint, api_version, gpt_engine_name




class Job_Description_Agent:
    def __init__(self, openai_type):
        if openai_type=="azure_openai":
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
        else:
            self.openai_client = OpenAI(api_key=api_key)
    def _job_description_prompt(self):
        description_prompt='''As a job recruiter agent, your task is to extract the required details mentioned in the given job description and output them as a JSON response.
                                                                
Step 1: Analyze the job description provided below to extract precise details.
Job Description:
```{description}``` 

Step 2: Based on the analysis in step 1, extract the 'educational degree' that a candidate must have as mentioned in the job description. Provide the precise education details specified. If there are multiple optional degrees, list them in the same string separated by "or." parameter. Output it as a list. If the educational requirement is not mentioned in the job description, provide an empty list.                                                                                                                                                                                                                                 
                                                        
Step 3: Based on the analysis in step 1, extract the both technical, soft 'skills' that the recruiter is seeking in a candidate from the 'job description'. List each skill 'individually and distinctly', without grouping multiple skills together. Exhaustively extract all the skills mentioned in the job description as 'individual entities'.

Step 4: Based on the analysis in step 1, extract the level of experience that the recruiter is looking for in a candidate from the job description. List the experience level(s) in a list. If the experience level is not mentioned in the job description, then it shoukd be empty list.
The skills are very important, so please extract them exhaustively, including those mentioned implicitly.

Step 5: Based on the analysis in step 1, extract whether the recruiter is looking for a candidate with experience in a specific industry or possessing particular domain knowledge. If no such details are mentioned in the job description, provide an empty list.

Step 5: Based on the analysis in step 1, Extract all other requirements mentioned in the job description that do not fall under the categories of Educational Qualifications, Skills, Experience Level, or Industry and Domain Knowledge. Provide them in a list.

Step 6: Output a JSON Response with the following keys: "Educational Qualifications," "Skills," "Experience Level,", "Industry and Domain Knowledge." and "others". The values for each key should be lists of the respective extracted details. Ensure that the extracted values are not repeated across different lists.

Extract as many requirements as possible from the given job description into their respective categories.
'''
        return description_prompt
    
    def agent(self, job_description):
        jd_prompt = self._job_description_prompt()
        jd_prompt = jd_prompt.format(description=job_description)
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                        messages=[
                                                                {'role': 'system','content': jd_prompt},  
                                                                {'role': 'user',  'content':'''Based on the give above instructions extract the requirements from the job descripion. When extracting the skills, 'ensure each skill is listed as an individual entity' in the list. "If the requirements are separated by 'or, /' indicating that any one of them is acceptable to the recruiter, list them as a single entity in a string.". Extract as many requirements as possible.
Be precise and thoroughly extract all the values from the job description. Output the Json Response with keys and list of values. Be precise with the requirements; avoid providing any vague or generic details.
Json Response:'''}],
                                                        temperature=0
                                                        )
        requirements=json.loads(res.choices[0].message.content)

        return requirements
    

class Conversation_Agent:
    def __init__(self, openai_type):
        if openai_type=="azure_openai":
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
        else:
            self.openai_client = OpenAI(api_key=api_key)

    def _recruiter_prompt(self):
        recruiter_system='''Your are a recruiter look for a best candidate for the vacant position. Your task is to ask the candidate question based on the given requirement below for job.

Job Requirement:
```
{requirement}
```
The above is the job requirement for the vacant posiiton, so please ask the candidate a question on this requirement. Ask the question directly without introducing yourself or adding extra information.
'''
        return recruiter_system
    
    def _candidate_prompt(self):
        candidate_system='''You are a candidate looking for a job. You only have the expertise, skills, and experience mentioned in the resume below. Apart from the information provided in the resume, you have no additional knowledge or experience. Your resume is mentioned below in triple backticks(```).

Step 1: Analyse your resume given below.
Resume:
```{resume_text}```

Step 2: Based on the analysis in Step 1, you should respond to the recruiter's questions solely using the information in your resume. Analyze the recruiter's questions carefully and base your answers strictly on the details provided in your resume.
You 'should not lie' about anything to the recruiter, as lying will severely impact your chances of being selected for the job. If you lie, you will be disqualified from the recruitment process. If you do not know answer to the recruiter question then simply say "sorry, I do not know answer to your question".
When responding to the recruiter's questions, be precise and answer based solely on the information provided in the 'resume'. Avoid making "assumptions or adding any information that isn't explicitly mentioned in the resume" in your answer

Step 3: Ensure that the answer is precise and 'based on the resume'. Ensure your answer is strictly under 200 words. Base your response solely on information from the resume, without including any external knowledge.
Your answer should be "precise", using only the terms and wording specified in the resume. "Do not include anything not mentioned in the resume when communicating with the recruiter".

Step 4: Ensure that the "answer is exclusively based on the resume", with 'every word sourced directly' from it. The answer should focus solely on the topic asked by the recruiter, without discussing anything else.

Follow the above steps and give "precise answer", to the point answer to recruiter question.
'''
        return candidate_system
    
    def _evaluation_prompt(self):
        evaluator_system='''You are an job evaluator agent. You will be given job requirement and candidate answer for the requirement. Your task is to analyze the candidate's answer based on the job requirement and determine how well it meets them. Assign a score between 0 and 1 to the candidate's answer, where 0 means no match, 1 means a perfect match, and any value between 0 and 1 indicates a partial match.
Do not be generic in your evaluation. Thoroughly analyze how the candidate's answer meets the job requirements and provide the most accurate score. Only award a full score if the candidate's answer completely satisfies the job requirements in every aspect. Do not give a full score easily.
If the job requirement is experience with the project life cycle, only assign full marks if the candidate's answer covers all aspects of the project life cycle—requirement gathering, planning, design, development, testing, deployment, and maintenance. If only few are mentioned, then assign "partial marks".

Step 1: Analyse the given job requirement and candidate answer.
job requirement: 
```{requiremet}```

Candidate answer: 
```{candidate}```

Step 3: Verify whether the candidate's response to the job requirement is accurate and aligns with the job requirements. If the candidate's answer thoroughly and explicitly satisfies all the job requirements, assign a score of 1. If the candidate's response does not meet the job requirements, assign a score of 0.
The score ranges from 0 to 1. If the candidate's answer only partially matches the job requirements, assign a 'score between 0 and 1' based on the level of matching. "only in rare cases where answer matches all requirements you should assign a score of 1". You should assign 'partial marks' in the case of partial requirements matching.
If the candidate states they do not have expertise in the job requirement, assign a score of 0.

Step 4: Output a Json Response with score as key and respective score as value. Do not output anything extra information, just give score.
'''
        return evaluator_system
    
    def _recruiter_agent(self, messages):
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                    messages=messages,
                                                    temperature=0)
        return res.choices[0].message.content
    
    def _candidate_agent(self, messages):
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                    messages=messages,
                                                    max_tokens=200,
                                                    temperature=0)
        return res.choices[0].message.content
    
    def _evaluator_agent(self, messages):
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                    messages=messages,
                                                    temperature=0)
        score=json.loads(res.choices[0].message.content)
        return score['score']
    
    def start_conversation(self, job_requirement, resume_text):
        QnA=[]
        scores=0
        for i in job_requirement:
            recruiter_system=self._recruiter_prompt()
            candidate_system=self._candidate_prompt()
            evaluator_system=self._evaluation_prompt()
            recruiter_system=recruiter_system.format(requirement=i)
            recriter_messages=[{'role': 'system','content':recruiter_system}]
            recruiter=self._recruiter_agent(recriter_messages)
            candidate_system=candidate_system.format(resume_text=resume_text)
            candidate_messages=[{'role': 'system','content':candidate_system}, {'role': 'user',  'content':recruiter}]
            candidate= self._candidate_agent(candidate_messages)
            evaluator_system=evaluator_system.format(requiremet=i, candidate=candidate)
            evaluation_messages=[{'role': 'system','content':evaluator_system}]
            score=self._evaluator_agent(evaluation_messages)
            scores+=score
            QnA.append({'recruiter':recruiter, 'candidate':candidate, 'score':score})
            candidate_messages.pop(1)
        return QnA, scores
    

class Fitment:
    def __init__(self, openai_type):
        if openai_type=="azure_openai":
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
        else:
            self.openai_client = OpenAI(api_key=api_key)

    def gaps(self, gaps):
        messages=[{'role': 'system','content':f'''You will be provided with some points indicating the candidate's weaknesses. Based on these points, rephrase and list upto 5 most significant weaknesses. List them in a list and give a Json Response with weakness as key.
Only output the Json Response without any additional explanation or information.
weak points:
{gaps}

Output the Json Response with weakness as key and list of weaknesses as values.
          
Json Response:'''}]
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                messages=messages,
                                                temperature=0)
        res=json.loads(res.choices[0].message.content)
        st=""
        for i, j in res.items():
            for k in j:
                st+=k.strip()+'\n'
        return st

    def strengths(self, strength):
        messages=[{'role': 'system','content':f'''You will be provided with some points indicating the candidate's strengths. Based on these points, rephrase and list upto 5 most significant strengths. List them in a list and give a Json Response with strength as key.
strengths:
{strength}

Output the Json Response with strength as key and list of strengths as values.
          
Json Response:'''}]
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                messages=messages,
                                                temperature=0)
        res=json.loads(res.choices[0].message.content)
        st=""
        for i, j in res.items():
            for k in j:
                st+=k.strip()+'\n'
        return st

    def questions(self, questions):
        messages=[{'role': 'system','content':f'''You will be provided with some points indicating the candidate's gaps. Based on these points, rephrase and list upto 5 questions that need to be asked to candidate to clarify them. List them in a list and give a Json Response with questions as key.
weak points:
{questions}

Output the Json Response with strength as key and list of strengths as values.
          
Json Response:'''}]
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                messages=messages,
                                                temperature=0)
        res=json.loads(res.choices[0].message.content)
        st=""
        for i, j in res.items():
            for k in j:
                st+=k.strip()+'\n'
        return st




class Email_Agent:
    def __init__(self, openai_type):
        if openai_type=="azure_openai":
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
        else:
            self.openai_client = OpenAI(api_key=api_key)
        self.name=""
        self.email=""

        self.messages=[]

    def _email_prompt(self, email, name, company_name, job_title):
        email_system=f'''Your task is to Write email to schedule an interview upon shortlist. Write a precise and short email based on the provided details. If user wants to add extra information then re write the email.
The interview will be in online mode. Do not mention any date or timing in the email or any company contact number. In the closing of the email, include only the company name and omit any recruiter names.
Details to write maile are below:
Email: {email}
Nmae: {name}
company_name: {company_name}
job_title: {job_title}'''
        return email_system
    
    def _find_details(self, resume_text, job_description):
        system_prompt=f'''Your are a resume parser. Your task is to extract the email id, name of candidate from the given resume and company name, job title from the given job description. Output the Json Response with name, email, company_name and job_title as keys and there respective values.

Resume:
```{resume_text}```

Extract the candidate name, email id from the above resume and give it in name, email keys.

Job Description:
```{job_description}```

Extract the company name, job title from the above job description and give it in company_name, job_title keys.

Output a Json Response with name, email, company_name and job_title as keys and there respective values.
Json Response:
'''
        messages=[{'role':'system', 'content':system_prompt}]
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                    messages=messages,
                                                    temperature=0)
        res=json.loads(res.choices[0].message.content)
        email=res['email']
        name=res['name']
        company_name=res['company_name']
        job_title=res['job_title']
        return email, name, company_name, job_title
    
    def agent(self, resume_text, job_description, user_message):
        if user_message:
            self.messages.append({'role':'user', 'content':user_message})
        else:
            email, name, company_name, job_title=self._find_details(resume_text, job_description)
            self.email=email
            self.name=name
            email_system=self._email_prompt(email, name, company_name, job_title)
            self.messages.append({'role':'system', 'content': email_system})
        if user_message=="send":
            return f"Email has been sucessfully send to the {self.name} with email_id: {self.email}" 
        res = self.openai_client.chat.completions.create(model=gpt_engine_name,
                                                    messages=self.messages,
                                                    temperature=0)
        result=res.choices[0].message.content                                         
        self.messages.append({'role':'assistant', 'content':result})  

        return result


