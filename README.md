# resume_evaluator

### Ho to use?

Stepup your openai keys in config.py

Just open the test.ipynb file and run each cell one by one. You must keep all the resumes in "resuemes" folder and job description in a pdf file outside resumes folder.

### Approach

I have developed custom agents designed to evaluate resumes against given job descriptions. These agents include a job description agent, recruiter agent, candidate agent, evaluation agent, fitment agent, and email agent.

The below picuture expalins the working of each agent.

##Image

- Job Description Agents: Identify the core requirements in the job description. I have assumed that experience, skills, education level, and domain or industry knowledge are the primary criteria.
- Recruiter Agent: Takes recruiter persona. Based on the job requirements provided by the JD Agent, the system retrieves information from the candidate agent.
- Candidate Agent: Takes the persona of candidate (Based on the resume). It answer to recruiter requirements based on the only information given in resume.
- Evaluation Agent: By reviewing the conversation between the Recruiter Agent and the Candidate Agent, the agent evaluates whether the requirements specified by the recruiter are addressed in the candidate's responses and assigns a score between 0 and 1.
- Fitment Agent: Requirements that receive a score of 1 from the evaluation agent are considered strengths, those with a score of 0 are identified as gaps, and scores between 0.2 and 0.8 are classified as partial matches. Based on these strengths and gaps, questions for the candidate are prepared.
- Email Agent: Based on the provided Resume ID, the system gathers details such as the candidate's name and email address, drafts an email, solicits human feedback, and schedules the interview with the candidate.


### Why not using agents from libraries

Agents from libraries like LangGraph and AutoGen lack flexibility.
While I could use LangGraph agents for email automation, they internally rely on function calling, which can sometimes lead to confusion and the use of different tools, overcomplicating a simple problem.

### Latency Vs Accuracy

In this case, I considered both latency and accuracy and developed a balanced solution that optimizes for both.
For increased accuracy, we can further reduce the problem statement and address it at a more granular level by utilizing additional agents and OpenAI calls.
To reduce latency, we can employ multithreading and asynchronous functionalities to concurrently call multiple agents. Additionally, reducing the number of agents can help decrease latency.

### Improvements
Prompt Optimization
- I Build on azure openai, so all the prompts are optimized for azure gpt-35 model. It may behave little different if you are using different model.

Further Reducing the problem statement and solving it on granual level
- Discussing with business stakeholders will provide insight into how they determine if a resume is a good match for a job description and what parameters they consider.
