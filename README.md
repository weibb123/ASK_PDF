# ASK_PDF


![image](https://github.com/weibb123/GPT-DocuAssist-/assets/84426364/5453a36d-d0fd-4c02-aec7-ca63a299214e)

tech stack used: langchain, openai, streamlit You can find versions in <b> requirements.txt </b>


## Steps to run this repo
git clone this repo
stream run app.py

## Motivation
People often receive many emails, daily of course. I was wondering if LLM sysytem could automate on this process by providing quality feedback.
Also after some research, people really do spent lots of times on emailing(business related).

https://hbr.org/2019/01/how-to-spend-way-less-time-on-email-every-day

<Goal>
  
Create a system that is capable of drafting responses to common business emails, such as customer inquiries or partnership requests.





## Evaluation
Multiple success metrics were considered: Track the reduction in response time, increase in email handling capacity, and customer satisfaction with email interactions.

For this project, I measure the time saved from using GPT compare to reading and responding on your own while maintaining a good quality.

Idea: Comparing the time for reading + responding email on your own and the time LLM generate a response.


On average, it saves about 32% of the time reading and writing emails.

## Lesson Learned
GPT-4-preview-1106 provide most reasoning and quality response due to updated knowledge (2023)

GPT-4 still good

GPT-3.5.turbo-1106 is good but sometimes giving extra informations that are not relevant.

Potentially: embed in email applications?

