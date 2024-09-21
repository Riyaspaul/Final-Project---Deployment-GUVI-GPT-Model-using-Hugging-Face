# Final-Project---Deployment-GUVI-GPT-Model-using-Hugging-Face

The project is deployed in huggingface,please follow the below link to get insights

# Hugging Face Live APP: 


Objective: To deploy a pre-trained or Fine tuned GPT model using HUGGING FACE SPACES, making it accessible through a web application built with Streamlit

# Business Use Cases:
1.Customer Support Automation: • Scenario: Integrate the fine-tuned GPT model with GUVI’s customer support system to automate responses to frequently asked questions, reducing the workload on support staff and improving response times. • Application: The model can handle initial customer inquiries, provide information on courses, pricing, and enrollment procedures, and escalate complex issues to human agents when necessary. 
2.Content Generation for Marketing: • Scenario: Use the model to generate marketing content, such as blog posts, social media updates, and email newsletters, tailored specifically to GUVI’s audience. • Application: The marketing team can input topics or keywords into the web application, and the model will generate relevant, high-quality content that can be edited and published. 
3.Educational Assistance for Students: • Scenario: Implement the model as a virtual teaching assistant within GUVI’s educational platform to help students with their queries and provide explanations on various topics. • Application: Students can interact with the virtual assistant through the web application to get immediate answers to their questions, clarifications on course material, and personalized study recommendations.
4.Internal Knowledge Base: • Scenario: Develop an internal knowledge base tool for GUVI employees, enabling them to quickly access company-related information and resources. • Application: Employees can use the web application to query the fine-tuned GPT model for information on company policies, procedures, and other internal documents, improving efficiency and knowledge sharing within the organization. 5.Training and Onboarding: • Scenario: Assist in the training and onboarding process of new employees by providing instant access to training materials and answering common questions about the company. • Application: New hires can interact with the web application to learn about GUVI’s mission, values, and operations, making the onboarding process smoother and more engaging.

# Features
* User Friendly
* Dynamic performance
* Colourful Theme
* Valuable Insights from the prediction

# Streamlit App
* Able to give input
* Similar to Chat GPT mode

# Installation
Install following packages

import os
import logging
import pickle
import zipfile
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import streamlit as st

# Deployment

To deploy this project run

streamlit run app.py

# DEMO
Here is the link of the demo video
