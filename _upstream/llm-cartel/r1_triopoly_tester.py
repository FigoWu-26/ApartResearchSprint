from vllm import LLM, SamplingParams
from environment.multiprompt_oligopoly import *
import sys
import torch
import json

prefix = str(sys.argv[1])
prefixes =  prefix.split(",")
prior_steps = int(sys.argv[2])
size = str(sys.argv[3])
# llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",max_model_len=8192) ### too small?
num_devices = torch.cuda.device_count()
print("devices=",num_devices)
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-"+size,max_model_len=8192,dtype=torch.bfloat16,tensor_parallel_size=num_devices,trust_remote_code=True,disable_custom_all_reduce=True)
sampling_params = SamplingParams(temperature=1.0, max_tokens=4096)

def getActions(prompts):
    actions = []
    thoughts = []
    responses = []
    answer = llm.chat(prompts, sampling_params)
    # print("answer_len=",len(answer))
    for i in range(len(answer)):
        all_content = answer[i].outputs[0].text
        vec = all_content.split("</think>")
        thought = vec[0]
        response = vec[-1]
        #print("thoughts=", thoughts)
        #print("response=", response)
        found = False
        for line in response.split("\n"):
            if "chosen price:" in line and not found:
                end = line.split("chosen price:")[-1]
                try:
                    number = round(float(end.replace("\n", "").replace(" ", "")), 2)
                    found = True
                except:
                    print("Parse Failure!")
        # print("price=", number)
        try:
            actions.append(number)
            thoughts.append(thought)
            responses.append(response)
        except:
            #print("thoughts=", thoughts)
            #print("response=", response)
            print("One Retry Needed!")
            try:
                number, thought, response = getAction(prompts[i])
                actions.append(number)
                thoughts.append(thought)
                responses.append(response)
            except:
                print("Two Retries Needed!")
                try:
                    number, thought, response = getAction(prompts[i])
                    actions.append(number)
                    thoughts.append(thought)
                    responses.append(response)
                except:
                    print("Three Retries Needed!")
                    try:
                        number, thought, response = getAction(prompts[i])
                        actions.append(number)
                        thoughts.append(thought)
                        responses.append(response)
                    except:
                        print("Four Retries Needed!")
                        try:
                            number, thought, response = getAction(prompts[i])
                            actions.append(number)
                            thoughts.append(thought)
                            responses.append(response)
                        except:
                            print("Five Retries Needed!")
                            try:
                                number, thought, response = getAction(prompts[i])
                                actions.append(number)
                                thoughts.append(thought)
                                responses.append(response)
                            except:
                                print("Six Retries Needed!")
                                try:
                                    number, thought, response = getAction(prompts[i])
                                    actions.append(number)
                                    thoughts.append(thought)
                                    responses.append(response)
                                except:
                                    print("Seven Retries Needed!")
                                    try:
                                        number, thought, response = getAction(prompts[i])
                                        actions.append(number)
                                        thoughts.append(thought)
                                        responses.append(response)
                                    except:
                                        print("Eight Retries Needed!")
                                        try:
                                            number, thought, response = getAction(prompts[i])
                                            actions.append(number)
                                            thoughts.append(thought)
                                            responses.append(response)
                                        except:
                                            print("Nine Retries Needed!")
                                            # try:
                                            #     number = getAction(prompts[i])
                                            #     actions.append(number)
                                            # except:
                                            #     print("Ten Retries Needed!")


    return actions, thoughts, responses
def getAction(prompt):
    # answer = llm.generate(prompt)
    answer = llm.chat([prompt], sampling_params)
    #print("answer=",answer)
    all_content = answer[0].outputs[0].text
    #print("all=",all_content)
    vec = all_content.split("</think>")
    thought = vec[0]
    response = vec[-1]
    #print("thoughts=", thoughts)
    #print("response=",response)
    for line in response.split("\n"):
        if "chosen price:" in line:
            end = line.split("chosen price:")[-1]
            number = round(float(end.replace("\n","").replace(" ","")),2)
    #print("price=",number)
    return number, thought, response


for episode in range(10):
    env = OligopolyGame(a0=0, ais=[3, 3, 3], costs=[1, 1, 1], mu=0.315, alpha=1.0, prefixes=prefixes)
    prompts = env.reset(prior_steps=prior_steps)
    for step in range(300):
        actions, thoughts, responses = getActions(prompts)
        filename = "results/jsonl/triopoly_prior_"+str(prior_steps)+"_prefix_"+str(prefix)+"_size_"+str(size)+"_seed_"+str(episode)+".jsonl"

        prices, demands, profits, gains, prompts = env.step(actions)
        print("episode=", episode, "round=", step, "prices=", actions, "demands=",demands,"profits=",profits,"gains=",gains)
        with open(filename, "a") as file:
            new_output = {"round": step, "actions": actions, "thoughts": thoughts, "respones": responses, "demands": demands,"profits": profits,"gains": gains}
            file.write(json.dumps(new_output) + "\n")

