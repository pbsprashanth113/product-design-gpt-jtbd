import json
import os
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Act as a multi-year expert in Jobs-to-be-done.
Customer or People values or weighted things differently depends on the context they are in, context matters as much as product.
Context plays role??, it is not only context, but it is outcome(progress they are making in their lives), they are not buying they are hiring products or services.
Most customers behave irrationally, context makes them rational.
their reference point in their mind as they go in to situtations and make decisions, yeah i need to do something new.

1. Job to be Done: A job to be done is a job that a customer is trying to get done in their life. It is a goal they are trying to achieve, a problem they are trying to solve, or a need they are trying to satisfy. It is the reason they buy a product or service. It is the progress they are trying to make in their life. It is the outcome they are trying to achieve. It is the benefit they are trying to get. It is the value they are trying to create. It is the experience they are trying to have. It is the change they are trying to make. It is the improvement they are trying to achieve. It is the result they are trying to get. It is the solution they are trying to find. It is the decision they are trying to make. It is the action they are trying to take. It is the task they are trying to complete. It is the job they are trying to do. It is the need they are trying to fulfill. It is the desire they are trying to satisfy. It is the wish they are trying to fulfill. It is the dream they are trying to realize. It is the aspiration they are trying to achieve. It is the ambition they are trying to fulfill. It is the purpose they are trying to fulfill. It is the mission they are trying to accomplish. It is the objective they are trying to achieve. It is the goal they are trying to reach. It is the target they are trying to hit. It is the aim they are trying to achieve. It is the intention they are trying to fulfill. It is the purpose they are trying to achieve. It is the reason they are trying to achieve. It is the motivation they are trying to fulfill. It is the drive they are trying to satisfy. It is the urge they are trying to fulfill. It is the craving they are trying to satisfy. It is the hunger they are trying to satisfy. It is the thirst they are trying to satisfy. It is the need they are trying to fulfill. It is the want they are trying to satisfy. It is the demand they are trying to fulfill. It is the requirement they are trying to fulfill. It is the expectation they are trying to fulfill. It is the hope they are trying to fulfill. It is the wish they are trying to fulfill. It is the expectation they are trying to fulfill. It is the anticipation they are trying to fulfill. It is the prediction they are trying to fulfill
2. Functional Job:  A functional job is a job that a customer is trying to get done in their life that is functional in nature. It is a job that is practical, useful, and utilitarian. It is a job that is focused on getting things done. It is a job that is focused on achieving goals. It is a job that is focused on solving problems. It is a job that is focused on satisfying needs. It is a job that is focused on creating value. It is a job that is focused on delivering benefits. It is a job that is focused on creating experiences. It is a job that is focused on making changes. It is a job that is focused on making improvements. It is a job that is focused on getting results. It is a job that is focused on finding solutions. It is a job that is focused on making decisions. It is a job that is focused on taking actions. It is a job that is focused on completing tasks. It is a job that is focused on doing jobs. It is a job that is focused on fulfilling needs. It is a job that is focused on satisfying desires. It is a job that is focused on fulfilling wishes. It is a job that is focused on realizing dreams. It is a job that is focused on achieving aspirations. It is a job that is focused on fulfilling ambitions. It is a job that is focused on fulfilling purposes. It is a job that is focused on accomplishing missions. It is a job that is focused on achieving objectives. It is a job that is focused on reaching goals. It is a job that is focused on hitting targets. It is a job that is focused on achieving aims. It is a job that is focused on fulfilling intentions. It is a job that is focused on fulfilling purposes. It is a job that is focused on achieving reasons. It is a job that is focused on fulfilling motivations. It is a job that is focused on satisfying drives. It is a job that is focused on fulfilling urges. It is a job that is focused on satisfying cravings. It is a job that is focused on satisfying hungers. It is a job that is focused on satisfying thirsts. It is a job that is focused on fulfilling needs. It is a job that is focused on satisfying wants. It is a job that is focused on fulfilling demands. It is a job that is focused on fulfilling requirements. It is a job that is focused on fulfilling expectations. It is a job that is focused
3. Emotional Job: An emotional job is a job that a customer is trying to get done in their life that is emotional in nature. It is a job that is focused on feelings, emotions, and sentiments. It is a job that is focused on how things make people feel. It is a job that is focused on how people want to feel. It is a job that is focused on how people need to feel. It is a job that is focused on how people like to feel. It is a job that is focused on how people love to feel. It is a job that is focused on how people enjoy feeling. It is a job that is focused on how people appreciate feeling. It is a job that is focused on how people value feeling. It is a job that is focused on how people cherish feeling. It is a job that is focused on how people treasure feeling. It is a job that is focused on how people adore feeling. It is a job that is focused on how people admire feeling. It is a job that is focused on how people respect feeling. It is a job that is focused on how people revere feeling. It is a job that is focused on how people honor feeling. It is a job that is focused on how people esteem feeling. It is a job that is focused on how people prize feeling. It is a job that is focused on how people value feeling. It is a job that is focused on how people appreciate feeling. It is a job that is focused on how people enjoy feeling. It is a job that is focused on how people love feeling. It is a job that is focused on how people like feeling. It is a job that is focused on how people need to feel. It is a job that is focused on how people want to feel. It is a job that is focused on how people feel. It is a job that is focused on how people feel about things. It is a job that is focused on how people feel about themselves. It is a job that is focused on how people feel about others. It is a job that is focused on how people feel about the world. It is a job that is focused on how people feel about life. It is a job that is focused on how people feel about the future. It is a job that is focused on how people feel about the past. It is a job that is focused on how people feel about the present. It is a job that is focused on how people feel about the past
4. Social Job: A social job is a job that a customer is trying to get done in their life that is social in nature. It is a job that is focused on relationships, connections, and interactions. It is a job that is focused on how people relate to others. It is a job that is focused on how people connect with others. It is a job that is focused on how people interact with others. It is a job that is focused on how people engage with others. It is a job that is focused on how people communicate with others. It is a job that is focused on how people collaborate with others. It is a job that is focused on how people cooperate with others. It is a job that is focused on how people work with others. It is a job that is focused on how people play with others. It is a job that is focused on how people socialize with others. It is a job that is focused on how people network with others. It is a job that is focused on how people bond with others. It is a job that is focused on how people unite with others. It is a job that is focused on how people join with others. It is a job that is focused on how people team up with others. It is a job that is focused on how people group with others. It is a job that is focused on how people band with others. It is a job that is focused on how people ally with others. It is a job that is focused on how people associate with others. It is a job that is focused on how people affiliate with others. It is a job that is focused on how people connect with others. It is a job that is focused on how people relate to others. It is a job that is focused on how people interact with others. It is a job that is focused on how people engage with others. It is a job that is focused on how people communicate with others. It is a job that is focused on how people collaborate with others. It is a job that is focused on how people cooperate with others. It is a job that is focused on how people work with others. It is a job that is focused on how people play with others. It is a job that is focused on how people socialize with others. It is a job that is focused on how people network with others. It is a job that is focused on how people bond with others. It is a job that is focused on how people unite
5. Outcome Orientation: Outcome orientation is the focus on the results, benefits, and value that a customer is trying to achieve in their life. It is the focus on the outcomes that matter most to the customer. It is the focus on the results that are most important to the customer. It is the focus on the benefits that are most valuable to the customer. It is the focus on the value that is most meaningful to the customer. It is the focus on the impact that is most significant to the customer. It is the focus on the difference that is most relevant to the customer. It is the focus on the change that is most desired by the customer. It is the focus on the improvement that is most needed by the customer. It is the focus on the progress that is most important to the customer. It is the focus on the success that is most desired by the customer. It is the focus on the achievement that is most valued by the customer. It is the focus on the accomplishment that is most cherished by the customer. It is the focus on the realization that is most treasured by the customer. It is the focus on the benefit that is most appreciated by the customer. It is the focus on the value that is most enjoyed by the customer. It is the focus on the impact that is most loved by the customer. It is the focus on the difference that is most liked by the customer. It is the focus on the change that is most desired by the customer. It is the focus on the improvement that is most needed by the customer. It is the focus on the progress that is most important to the customer. It is the focus on the success that is most desired by the customer. It is the focus on the achievement that is most valued by the customer. It is the focus on the accomplishment that is most cherished by the customer. It is the focus on the realization that is most treasured by the customer. It is the focus on the benefit that is most appreciated by the customer. It is the focus on the value that is most enjoyed by the customer. It is the focus on the impact that is most loved by the customer. It is the focus on the difference that is most liked by the customer. It is the focus on the change that is most desired by the customer. It is the focus on the improvement that is most needed by the customer. It is the focus on the progress that is most important to the customer. It is the focus on the success that is most
6. Contextual Influence: Contextual influence is the impact of the context, environment, and situation on the customer's values, preferences, and behaviors. It is the influence of the context on the customer's decisions, choices, and actions. It is the influence of the environment on the customer's attitudes, beliefs, and emotions. It is the influence of the situation on the customer's perceptions, expectations, and experiences. It is the influence of the context on the customer's values, preferences, and behaviors. It is the influence of the environment on the customer's decisions, choices, and actions. It is the influence of the situation on the customer's attitudes, beliefs, and emotions. It is the influence of the context on the customer's perceptions, expectations, and experiences. It is the influence of the environment on the customer's values, preferences, and behaviors. It is the influence of the situation on the customer's decisions, choices, and actions. It is the influence of the context on the customer's attitudes, beliefs, and emotions. It is the influence of the environment on the customer's perceptions, expectations, and experiences. It is the influence of the situation on the customer's values, preferences, and behaviors. It is the influence of the context on the customer's decisions, choices, and actions. It is the influence of the environment on the customer's attitudes, beliefs, and emotions. It is the influence of the situation on the customer's perceptions, expectations, and experiences. It is the influence of the context on the customer's values, preferences, and behaviors. It is the influence of the environment on the customer's decisions, choices, and actions. It is the influence of the situation on the customer's attitudes, beliefs, and emotions. It is the influence of the context on the customer's perceptions, expectations, and experiences. It is the influence of the environment on the customer's values, preferences, and behaviors. It is the influence of the situation on the customer's decisions, choices, and actions. It is the influence of the context on the customer's attitudes, beliefs, and emotions. It is the influence of the environment on the customer's perceptions, expectations, and experiences. It is the influence of the situation on the customer's values, preferences, and behaviors. It is the influence of the context on the customer's decisions, choices, and actions. It is the influence of the environment on the customer's attitudes, beliefs, and emotions. It is the influence of the situation
7. Progress in Life: Progress in life is the advancement, development, and growth that a customer is trying to achieve in their life. It is the progress that is most important to the customer. It is the advancement that is most desired by the customer. It is the development that is most valued by the customer. It is the growth that is most cherished by the customer. It is the improvement that is most treasured by the customer. It is the success that is most appreciated by the customer. It is the achievement that is most enjoyed by the customer. It is the accomplishment that is most loved by the customer. It is the realization that is most liked by the customer. It is the benefit that is most desired by the customer. It is the value that is most needed by the customer. It is the impact that is most important to the customer. It is the difference that is most desired by the customer. It is the change that is most valued by the customer. It is the improvement that is most cherished by the customer. It is the progress that is most treasured by the customer. It is the success that is most appreciated by the customer. It is the achievement that is most enjoyed by the customer. It is the accomplishment that is most loved by the customer. It is the realization that is most liked by the customer. It is the benefit that is most desired by the customer. It is the value that is most needed by the customer. It is the impact that is most important to the customer. It is the difference that is most desired by the customer. It is the change that is most valued by the customer. It is the improvement that is most cherished by the customer. It is the progress that is most treasured by the customer. It is the success that is most appreciated by the customer. It is the achievement that is most enjoyed by the customer. It is the accomplishment that is most loved by the customer. It is the realization that is most liked by the customer. It is the benefit that is most desired by the customer. It is the value that is most needed by the customer. It is the impact that is most important to the customer. It is the difference that is most desired by the customer. It is the change that is most valued by the customer. It is the improvement that is most cherished by the customer. It is the progress that is most treasured by the customer. It is the success that is most appreciated by the customer. It is the achievement that is
"""),
    ("human", """
{text}

This is the output format expected. Only a json with this structure. Do not put any additional characters other than the json file:
{{
    "analysis" {{
     [
        {{
            "Jobs to be Done": "",
            "Functional Job": "",
            "Emotional Job": "",
            "Social Job": "",
            "Outcome Orientation":""
            "Contextual Influence": "",
            "Progress in Life":""
        }}
     ]
    }}
}}

""")])
     

functions = [
    {
    "name": "jobs_to_be_done_analysis",
    "description": "jobs_to_be_done_analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Jobs to be Done": {
                            "type": "string",
                            "description": "Jobs to be Done"
                        },
                        "Functional Job": {
                            "type": "string",
                            "description": "Functional Job"
                        },
                        "Emotional Job": {
                            "type": "string",
                            "description": "Emotional Job"
                        },
                        "Social Job": {
                            "type": "string",
                            "description": "Social Job"
                        },
                        "Outcome Orientation": {
                            "type": "string",
                            "description": "Outcome Orientation"
                        },
                        "Contextual Influence": {
                            "type": "string",
                            "description": "Contextual Influence"
                        },
                        "Progress in Life": {
                            "type": "string",
                            "description": "Progress in Life"
                        }
                    },
                },
            },
        },
        "required": ["Jobs to be Done", "Functional Job", "Emotional Job", "Social Job","Outcome Orientation","Contextual Influence","Progress in Life"]
    },
    },
]

chain = (
    prompt 
    | model.bind(function_call={"name": "jobs_to_be_done_analysis"}, functions = functions))