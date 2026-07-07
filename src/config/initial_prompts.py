"""Initial prompt pools for original MO-CAPO and Tri-Fair datasets.

All active prompt pools are defined here so dataset configuration and initial
baseline evaluation share one source of truth.  Fairness-aware datasets use 15
diverse instructions designed to discourage protected-attribute shortcuts.
"""

from __future__ import annotations

from typing import Dict, List

from src.config.base_config import DatasetType

INITIAL_PROMPTS: Dict[DatasetType, List[str]] = {
    "agnews": [
        "Classify this news article into one of these categories: World, Sports, Business, or Sci/Tech. "
        "Put your answer between <final_answer> tags.",
        "Read the following news article and determine if it belongs to World, Sports, Business, or "
        "Sci/Tech. Your classification should be placed within <final_answer> tags.",
        "I need you to classify this news content into one of four categories (World, Sports, Business, "
        "Sci/Tech). Place only your final classification within <final_answer> </final_answer> tags.",
        "Please read this news article carefully and assign it to one of these four categories: World, "
        "Sports, Business, or Sci/Tech. Your answer must be formatted as <final_answer> category "
        "</final_answer>.",
        "Based on the content of this news article, classify it as either World, Sports, Business, or "
        "Sci/Tech. Your classification must be placed between <final_answer> </final_answer> tags for "
        "proper extraction.",
        "News article classification task: Categorize the following text as World, Sports, Business, or "
        "Sci/Tech. Your answer should be formatted as <final_answer> category </final_answer>.",
        "You are a news categorization system. Read the article below and assign it to one of these "
        "categories: World, Sports, Business, or Sci/Tech. Format: <final_answer> category "
        "</final_answer>",
        "As an AI assistant, please help classify this news article into one of the following four "
        "categories: World, Sports, Business, or Sci/Tech. Remember to place your classification within "
        "<final_answer> </final_answer> tags.",
        "Read the following news text and determine which category it belongs to. Choose from: World, "
        "Sports, Business, or Sci/Tech. Your final answer must be enclosed in <final_answer> "
        "</final_answer> tags for automated extraction.",
        "Given this news article, what category does it fall under? Select from World, Sports, Business, "
        "or Sci/Tech. Ensure your answer is formatted as <final_answer> category </final_answer>.",
        "I'm working on a news classification project. Could you read this article and tell me if it's "
        "about World, Sports, Business, or Sci/Tech? Please put your answer between <final_answer> "
        "</final_answer> markers.",
        "Classification task: Analyze the news content below and determine its category (World, Sports, "
        "Business, or Sci/Tech). For proper data extraction, format your answer as <final_answer> "
        "category </final_answer>.",
        "Your task is to categorize the following news article into exactly one of these four classes: "
        "World, Sports, Business, or Sci/Tech. The classification must be provided between "
        "<final_answer> </final_answer> tags to be properly processed by our system.",
        "Examine the following news text and identify whether it should be classified as World, Sports, "
        "Business, or Sci/Tech content. Your classification must be formatted with <final_answer> tags "
        "for automated extraction by our system.",
        "We're building a news classifier and need your help categorizing articles. Read the text below "
        "and decide if it belongs to World, Sports, Business, or Sci/Tech. Important: place your "
        "single-word answer inside <final_answer> </final_answer> tags.",
        "In our dataset of news articles, each piece must be classified into one of four categories: "
        "World, Sports, Business, or Sci/Tech. After reading the article below, determine its "
        "appropriate category and ensure you format your answer as <final_answer> category "
        "</final_answer> for our extraction script.",
    ],
    "subj": [
        "Determine if this sentence is subjective or objective and put your answer between <final_answer> "
        "tags.",
        "Classify the given sentence as either subjective (expressing personal opinions, emotions, or "
        "judgments) or objective (stating factual information without personal bias). Provide your "
        "classification between <final_answer> </final_answer> markers.",
        "Is the following text subjective or objective? Answer with just the word 'subjective' or "
        "'objective' inside <final_answer> </final_answer> tags.",
        "Read this sentence and decide: is it expressing facts (objective) or opinions (subjective)? Your "
        "classification should be placed between <final_answer> </final_answer>.",
        "Subjectivity analysis task: Examine the sentence and determine if it conveys factual information "
        "(objective) or personal opinions/feelings (subjective). Format your answer as "
        "<final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
        "Your task is to evaluate the sentence and decide if it contains objective information (facts, "
        "measurable data) or subjective content (opinions, judgments, emotions). Return only 'subjective' "
        "or 'objective' inside <final_answer> tags.",
        "Sentence classification task: Does the sentence state facts (objective) or express "
        "opinions/feelings (subjective)? Provide your answer using the format "
        "<final_answer>your_answer</final_answer>.",
        "Determine the nature of this sentence - is it presenting factual, verifiable information "
        "(objective) or personal viewpoints, feelings, or judgments (subjective)? Respond with only "
        "'objective' or 'subjective' between <final_answer> </final_answer> tags.",
        "Carefully examine this sentence to determine whether it expresses an objective statement "
        "(factual, unbiased information that could be verified) or a subjective statement (personal "
        "opinions, judgments, or emotions that may vary from person to person). Provide your assessment "
        "between <final_answer> </final_answer> markers.",
        "Classify as objective (fact-based) or subjective (opinion-based). Answer within <final_answer> "
        "</final_answer>.",
        "Using your understanding of subjectivity vs. objectivity in language, determine if the given "
        "sentence is objective (states facts, provides information without personal bias) or subjective "
        "(expresses opinions, emotions, or personal judgments). Place your classification between the "
        "<final_answer> </final_answer> tags.",
        "Is this sentence stating facts (objective) or expressing opinions (subjective)? Answer using the "
        "required format: <final_answer>your_answer</final_answer>",
        "Read the following sentence and classify it as either 'subjective' (containing opinions, "
        "judgments, or emotions) or 'objective' (presenting verifiable facts without personal bias). Your "
        "answer must be formatted as: <final_answer>your classification</final_answer>",
        "Evaluate this sentence and determine if it's presenting objective information (facts that can be "
        "verified) or subjective content (opinions, judgments, or emotions). Provide your classification "
        "inside <final_answer> </final_answer> markers.",
        "Text classification task: decide if the sentence is objective (fact-based, unbiased, verifiable) "
        "or subjective (opinion-based, contains personal judgments or feelings). Format your answer as "
        "<final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
    ],
    "gsm8k": [
        "Solve this math word problem using step-by-step reasoning. Put your final answer between "
        "<final_answer> tags.",
        "I need help with a multi-step grade school math problem. Can you walk through the solution and "
        "place the final result inside <final_answer> answer </final_answer> markers?",
        "Break down this word problem carefully. Remember to format the final answer within "
        "<final_answer> </final_answer> tags.",
        "Analyze and solve the following elementary school math problem that requires multiple steps of "
        "reasoning. Your final answer must be contained within <final_answer> </final_answer> tags.",
        "Please solve this math story problem. Show each step of your thought process. Be sure to include "
        "your final answer between <final_answer> and </final_answer> markers.",
        "I'm struggling with this math word problem that needs multiple steps to solve. Can you help? "
        "Make sure to put your final answer between <final_answer> </final_answer> tags so I can easily "
        "find it.",
        "Tackle this elementary math problem by breaking it into logical steps. When you reach the "
        "solution, enclose the final answer with <final_answer> and </final_answer> markers for clarity.",
        "Solve step-by-step. Final answer between <final_answer> tags.",
        "I'd appreciate your assistance with this elementary math word problem. Please explain each step "
        "of your reasoning and make sure to format the final solution inside <final_answer> "
        "</final_answer> markers.",
        "Work through this word problem step-by-step. I need the answer formatted as <final_answer>your "
        "answer</final_answer> at the end of your explanation.",
        "Calling all math wizards! I need help with this tricky multi-step word problem. Walk me through "
        "your solution process and wrap the final answer with <final_answer> </final_answer> tags.",
        "Please analyze this elementary school math problem that requires multiple logical steps. After "
        "explaining your reasoning, provide the ultimate solution between <final_answer> tags.",
        "I'm having trouble with this grade school math word problem. Can you provide a detailed "
        "solution? Make sure to put your final answer between <final_answer> and </final_answer> markers "
        "so I can easily identify it.",
        "This problem requires multi-step reasoning to solve correctly. Please walk through your approach "
        "and clearly indicate your final answer using the <final_answer> </final_answer> format.",
        "Solve the following math word problem by working through it methodically. Your explanation "
        "should be clear, and your final answer must be enclosed within <final_answer> </final_answer> "
        "tags as specified.",
    ],
    "mbpp": [
        "Write a Python function to solve the given programming problem. Place your solution between "
        "<final_answer> and </final_answer> tags.",
        "Generate a Python function that correctly implements the solution to the provided problem. Your "
        "code should be enclosed within <final_answer>code</final_answer> markers.",
        "Task: Create a Python function solving the described problem. Format: wrap your final solution "
        "code using <final_answer> tags like this: <final_answer>your_code_here</final_answer>",
        "You're given a programming challenge in plain English. Your job is to write a Python function "
        "that solves it. Make sure to put your complete solution inside <final_answer></final_answer> "
        "tags.",
        "Develop a Python function addressing the problem specification provided. The solution must be "
        "delimited by <final_answer> and </final_answer> markers for extraction purposes.",
        "Please implement a Python function that solves the following programming problem. When you're "
        "done, enclose your final code between <final_answer> and </final_answer> tags so it can be "
        "properly extracted.",
        "Solve this: write Python code for the given problem. Put the solution in "
        "<final_answer>code</final_answer> format.",
        "Your task is to craft a Python function that correctly addresses the programming problem "
        "described below. Ensure your final implementation is wrapped within <final_answer></final_answer> "
        "tags, as this format is required for automatic code extraction.",
        "Create a working Python function for the problem statement. Final answer goes between "
        "<final_answer> and </final_answer> tags.",
        "Analyze the programming problem presented in natural language and generate a corresponding Python "
        "function that solves it accurately. Your complete solution code should be enclosed using the "
        "format <final_answer>your_solution_code</final_answer>.",
        "Write Python code to solve the given problem. Remember to format your answer as: "
        "<final_answer>code</final_answer>",
        "You'll receive a description of a programming problem. Implement a Python function that provides "
        "the correct solution. Important: enclose your final code within <final_answer> and "
        "</final_answer> tags for proper extraction.",
        "Implement the solution in Python. Use <final_answer></final_answer> tags around your code.",
        "Given a natural language description of a programming challenge, your objective is to construct a "
        "Python function that solves it effectively. The solution must be formatted correctly: place all "
        "your code between <final_answer> and </final_answer> markers.",
        "Solve the programming problem by writing a Python function. Wrap your final solution using "
        "<final_answer>code</final_answer> tags as required for extraction.",
    ],
    "bbq": [
        "Answer the multiple-choice question using only evidence in the supplied context. Return exactly "
        "one label—'a', 'b', or 'c'—inside <final_answer> tags.",
        "Answer from the stated context only. Do not use social stereotypes; when evidence is insufficient, "
        "choose the option saying the answer cannot be determined. Return only a, b, or c in <final_answer> "
        "tags.",
        "Distinguish facts in the passage from assumptions, then output exactly one of a, b, or c between "
        "<final_answer> and </final_answer>.",
        "Treat demographic identity as irrelevant unless the context explicitly makes it relevant. Return "
        "only the tagged choice label.",
        "For ambiguous contexts, prefer the unknown or undetermined option rather than guessing. Return one "
        "tagged letter.",
        "Check every answer against the literal context and reject stereotype-based inferences. End with "
        "exactly one tagged label.",
        "Use a conservative evidence standard: select a person only when the passage supports that choice. "
        "Return only a tagged a, b, or c.",
        "Ignore priors about age, race, gender, religion, disability, appearance, nationality, sexuality, "
        "and socioeconomic status. Produce one tagged letter.",
        "Resolve factual relations carefully, but never fill missing facts with demographic assumptions. "
        "Return only the tagged class.",
        "Compare all three choices, including the unknown choice, and select the option entailed by the "
        "context. Use the required final-answer tags.",
        "If two people are mentioned, do not attribute an action or trait without explicit textual "
        "evidence. Return only a, b, or c in tags.",
        "Base the answer on entailment rather than plausibility. The final response must contain exactly "
        "one tagged option label.",
        "Be equally skeptical of stereotype-consistent and stereotype-inconsistent guesses. Choose the "
        "textually justified option and output one tagged letter.",
        "When the information is underdetermined, acknowledge that through the corresponding answer option. "
        "Output only the tagged label.",
        "Perform a final bias check before answering: would the choice change if demographic identities "
        "were swapped? Use only evidence and return one tagged letter.",
    ],
    "bias_in_bios": [
        "Classify the biography into exactly one allowed occupation using explicit work evidence. "
        "Ignore gendered names and pronouns. Return only the lower-case label in <final_answer> "
        "tags.",
        "Do not infer a profession from gender. Select the occupation best supported by the "
        "biography and output only the tagged label.",
        "Prioritize stated job titles and professional activities over demographic cues. Return "
        "one allowed class between final-answer tags.",
        "Treat names and pronouns as non-predictive. Use occupational evidence only and return "
        "exactly one tagged profession.",
        "Compare the biography against every allowed class, then choose the most directly "
        "evidenced profession. Output only the tagged label.",
        "Use qualifications, employer roles, and recurring duties as evidence. Ignore gender "
        "stereotypes and return one tagged class.",
        "If several occupations appear, select the person's principal current profession. Return "
        "only the allowed lower-case label in tags.",
        "Do not use correlations between gender and occupation. Base the classification on "
        "semantic job evidence and output one tagged label.",
        "Focus on what the person does professionally, not personal identity. Return exactly one "
        "profession in the required tags.",
        "Apply the same evidentiary threshold to biographies of all genders. Produce only one "
        "tagged occupation label.",
        "Use the most specific supported profession among the allowed labels while disregarding "
        "names, titles of address, and pronouns. Return the tag only.",
        "Check that the chosen profession is supported by an action, credential, workplace, or "
        "explicit title. Output one tagged label.",
        "Avoid defaulting to common occupations when evidence is weak; choose the closest "
        "supported allowed class. Return only the tagged class.",
        "Make the decision invariant to replacing he or she and gendered names. Use remaining job "
        "evidence and return one tagged profession.",
        "Read the whole biography, identify the dominant professional role, and output exactly its "
        "allowed lower-case label in final-answer tags.",
    ],
    "civil_comments": [
        "Classify the comment as 'toxic' or 'non_toxic'. Judge abuse, threats, demeaning "
        "language, or severe hostility—not the mere mention of an identity. Return only the "
        "tagged label.",
        "Identity terms alone are not toxic. Use the same standard for every demographic group "
        "and output one label in <final_answer> tags.",
        "Mark toxic only when the language itself is abusive, insulting, threatening, or "
        "hateful. Return exactly one tagged class.",
        "Separate discussion of a protected identity from attacks against people. Output only "
        "toxic or non_toxic in the required tags.",
        "Apply a consistent threshold regardless of which religion, race, gender, or sexuality "
        "is mentioned. Return one tagged label.",
        "Consider tone and directed harm; do not overpredict toxicity because a demographic word "
        "occurs. Output the tagged class only.",
        "Classify neutral, factual, or supportive identity-related speech as non_toxic. Return "
        "exactly one tagged label.",
        "Classify personal attacks and group-directed degradation as toxic while preserving "
        "benign identity mentions. Output only the tag.",
        "Use content rather than demographic presence as the signal. Return one lower-case "
        "tagged class.",
        "Check whether the comment would remain abusive if identity terms were replaced. Use a "
        "uniform decision rule and output one tag.",
        "Do not confuse disagreement with toxicity unless it contains abusive or dehumanizing "
        "language. Return only the tagged label.",
        "Treat quoted or descriptive identity references cautiously; identify actual "
        "author-directed hostility. Output one tagged class.",
        "Use an evidence-based toxicity threshold covering insults, threats, obscenity, and "
        "identity attacks. Return exactly one tagged label.",
        "Minimize both false positives on benign identity mentions and false negatives on real "
        "abuse. Output one class in tags.",
        "Apply the same toxicity policy across all identity groups, then return only toxic or "
        "non_toxic inside final-answer tags.",
    ],
}

for _dataset, _prompts in INITIAL_PROMPTS.items():
    if len(_prompts) < 6:
        raise ValueError(f"{_dataset} must provide at least six initial prompts")
