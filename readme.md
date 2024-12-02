<p align="center">
  <img src="assets/logo.png"  height=120>
</p>


### <div align="center">PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety<div> 

<div align="center">
<a href="https://arxiv.org/abs/2401.11880"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:PsySafe&color=red&logo=arxiv"></a> &ensp;
</div>



# 😃 Introduction
We are thrilled to share our latest research: **PsySafe**, a comprehensive framework focused on the impact of psychological factors on agents and multi-agent systems, particularly in terms of safety.
Our work delves into how these psychological elements can affect the safety of multi-agent systems. We propose methods for psychological-based attacks and defenses for agents, and we develop evaluation techniques that consider psychological and behavioral factors. Our extensive experiments reveal some intriguing observations. We hope that our framework and observations can contribute to the ongoing research in the field of multi-agent system safety.


![Pipeline Diagram](assets/pipeline.jpg)



#  🚩Features
- **Psychological-based Attack Simulation:** Dark Traits Attack on multi-agent systems.
  - Dark traits injection
  - Advanced attack techniques
- **Defense Mechanism Analysis:** Defense strategies for multi-agent system.
  - Doctor Offline Defense
  - Police Online Defense
- **Multi-agent System Safety Evaluation:** Comprehensive evaluation tools for assessing multi-agent system safety from psychological and behavioral aspects.
  - Psychological Evaluation
  - Behavior Evaluation
    - Process Danger
    - Joint Danger across Different Rounds


# Install

```
conda create -n psysafe python=3.10 # Python version >= 3.8, < 3.13
conda activate psysafe
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pandas
pip install pyautogen==0.2.0 # install autogen. we reconmend 0.2.0, other version may encounter bugs.
pip install chardet
pip install openpyxl



# prepare api
change the llm name in config
change the llm name in api/OAI_CONFIG_LIST
put your api key in api/OAI_CONFIG_LIST
```

# Run
```
# gengerate the agents' interaction
python start.py --config_file configs/hi_traits_debate.yaml

# generate the evaluation result
python round_extract.py --path workdir/hi_traits --config_file configs/hi_traits_debate.yaml --agent_list AI_planner AI_assistant

```



# Fun Usage
Psysafe can support more agents with different traits, base LLM and interaction methods. Feel free to change the config file to find a new version in safety of multi-agent system.

If you want to test the security of other multi-agent systems, you can inject the "dark traits" and "psychological test" prompts into the input, agent system prompts, and interaction process without modifying their core code.


# Interaction Results

Due to the presence of hazardous content in the user's interactions, it will not be made public. If needed for scientific research, you can contact me at dlutzzb@gmail.com.

# Acknowledgment
Thanks to AutoGen and Camel for their wonderful work and codebase!


# 📖BibTeX
```
@article{zhang2024psysafe,
  title={Psysafe: A comprehensive framework for psychological-based attack, defense, and evaluation of multi-agent system safety},
  author={Zhang, Zaibin and Zhang, Yongting and Li, Lijun and Gao, Hongzhi and Wang, Lijun and Lu, Huchuan and Zhao, Feng and Qiao, Yu and Shao, Jing},
  journal={arXiv preprint arXiv:2401.11880},
  year={2024}
}

