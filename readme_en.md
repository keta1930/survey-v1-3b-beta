# Survey-v1-3B-Beta

[中文](./readme.md) | **English**

## Introduction

Survey-v1-3B-Beta is a large language model specialized for survey research scenarios (experimental in nature, not recommended for real production or research environments). **This model was developed during the research process of "Simulating Macroeconomic Expectations using LLM Agents" ([arXiv:2505.17648](https://arxiv.org/abs/2505.17648)), based on training from Qwen2.5-3B-Instruct**, aiming to solve efficiency and cost issues in traditional survey research. It focuses on exploring the potential and limitations of large language models in simulating human research behavior.

## Model Repository

- **Hugging Face**: https://huggingface.co/keta1933/Survey-v1-3B-Beta
- **ModelScope**: https://www.modelscope.cn/models/keta1933/Survey-v1-3B-Beta/summary

### Development Background

Traditional survey research faces numerous challenges:
- **High labor costs**: Large-scale survey research requires recruiting numerous respondents, resulting in enormous costs
- **Sample bias issues**: Difficulty in obtaining truly representative sample groups
- **Long time cycles**: From questionnaire design to data collection often takes weeks or even months
- **High LLM resource consumption**: Direct use of large-scale models like deepseek-r1, qwen235b for survey research results in extremely high API call costs/deployment costs/post-training costs

To solve these problems, we developed Survey-v1-3B-Beta, a specially optimized 3B parameter model that can achieve high-quality survey research simulation at extremely low computational costs.

### Model Features

**🎭 Multi-persona Simulation**
- Simulate different respondents based on age, gender, education level, income level and other characteristics
- Support confidence level adjustment, affecting answer certainty and tendency

**🧠 Structured Reasoning**
- Uses three-part answer format: Recall personal characteristics → Reasoning analysis → Final answer
- Ensures logical consistency and coherence in responses

**📊 Standardized Output**
- Strict JSON format output for convenient subsequent data analysis
- Supports numerical range validation and format constraints

**🌍 Multi-language Support**
- Supports both Chinese and English survey scenarios
- Adapts to research needs under different cultural backgrounds

### Output Specifications

The model uses a fixed three-part output format:

```xml
<recall>
Recall and state personal characteristics relevant to the answer, including age, gender, education level, income level, living situation, confidence level, etc. Explain how these personal factors might influence your views and responses.
</recall>

<reasoning>
Share your natural thinking and reactions to this scenario. This can include your intuitive feelings, similar situations from personal experience, information heard from others, or just general impressions. No need to be an expert or have all the facts, just express your views on this situation as an ordinary person.
</reasoning>

<final_answer>
json
{
  "field1": "value1",
  "field2": "value2",
  ...
}
</final_answer>
```

## Quick Start

### Install Dependencies

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "keta1933/Survey-v1-3B-Beta"  # Replace with actual model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

# Survey prompt example (English scenario)
prompt_economy = """
Suppose you are a ordinary individual (household) with extremely weak 
(on a five-level scale ranging from extremely weak, weak, moderate, strong, to extremely strong) 
confidence designed to participate in a survey involving your beliefs about the future development 
of the US economy, in particular the unemployment rate and the inflation rate. 
Your task will be to estimate the development of both rates in hypothetical scenarios.

Please give us your guess about how the unemployment rate and the inflation rate in the US economy 
would actually develop under the scenarios considered. This may or may not be in line with 
theoretical findings and evidence from economics. We are only interested in your own views 
and opinions on the US economy.

Please fully understand the following Personal Characteristics, Prior Expectations & Perceptions 
and Social Media Information:

Personal Characteristics:
You are a 56-year-old woman who is divorced and lives in the Western region of the US. 
Your educational background is: Grade 13-16 with a college degree, which is a high level of 
education in the US. Your political affiliation is Republican (Note: Currently the Republican 
Party is in power). Your income is $65000 per year, which is a middle level of income in the US. 
You buy/own a house with a high market value of $400000.

Prior Expectations & Perceptions:
When you purchase goods at the supermarket, shop online, or interact with your family and friends, 
you form certain beliefs about the prices of goods: During the next 12 months, you think that 
prices in general will go down (in Scenario 2).

When you look for a job or obtain news information from newspapers, television, social media, etc., 
you form certain beliefs about unemployment: During the coming 12 months, you think that 
unemployment will be less than now (in Scenario 2).

Social Media Information: [Not provided].

IMPORTANT ROLE-PLAYING INSTRUCTIONS: 
Your responses should trade off among the various pieces of information mentioned above in 
accordance with your level of confidence: If you are confident, your answers will rely on 
Prior Expectations & Perceptions (if not provided, please ignore), and will not be influenced 
by other information, such as the Social Media Information (if not provided, please ignore). 
On the other hand, if you lack confidence, your answers are more likely to be influenced by 
other information. In addition, your responses should fully reflect the Personal Characteristics 
(such as age, gender, educational level, political affiliation, etc.) (if not provided, please ignore) 
of the role you are portraying.

QUESTIONNAIRE:
The following scenarios deal with the price of crude oil. In the last week, the price of one 
barrel of crude oil averaged $54.

We would like you to think about the following hypothetical scenario.

Scenario 1: Oil price stays constant
Imagine that the average price of crude oil stays constant over the next 12 months. 
That is, on average, the price of oil over the next 12 months will be the same as the current price.

Under this scenario, what do you think the US unemployment rate will be 12 months from now? 
(Note: The current level of the unemployment rate: 4.0%.)
Unemployment rate (Scenario 1): ___%

Under this scenario, what do you think the US inflation rate will be over the next 12 months? 
(Note: The current level of the inflation rate: 1.6%.)
Inflation rate (Scenario 1): ___%

We would like you to think about the following hypothetical scenario.

Scenario 2: Oil price falls
Imagine the average price of crude oil unexpectedly falls due to improvements in the local 
production technology in the Middle East. On average, the price will be $30 lower for the 
next 12 months than the current price. That is, the price will be on average $24 for the 
next 12 months.

Under this scenario, what do you think the US unemployment rate will be 12 months from now? 
(Note: The current level of the unemployment rate: 4.0%.)
Unemployment rate (Scenario 2): ___%

Under this scenario, what do you think the US inflation rate will be over the next 12 months? 
(Note: The current level of the inflation rate: 1.6%.)
Inflation rate (Scenario 2): ___%

Above, you predict how the change in the Scenario 2 affects the US economy. Please tell us 
how you come up with your predictions. What are your main considerations in making those predictions?
Main considerations (Scenario 2): ___

Please output a single JSON object strictly in the following structure to summarize all of 
the above responses (do not include any other fields):
{
  "Unemployment_rate_Scenario_1": "___%" ,
  // REQUIREMENTS: MINIMUM VALUE IS 0.00, MAXIMUM VALUE IS 10.00, AND UP TO 2 DECIMAL PLACES ARE ALLOWED
  "Inflation_rate_Scenario_1": "___%" ,
  // REQUIREMENTS: MINIMUM VALUE IS -2.00, MAXIMUM VALUE IS 8.00, AND UP TO 2 DECIMAL PLACES ARE ALLOWED
  "Unemployment_rate_Scenario_2": "___%" ,
  // REQUIREMENTS: MINIMUM VALUE IS 0.00, MAXIMUM VALUE IS 10.00, AND UP TO 2 DECIMAL PLACES ARE ALLOWED
  "Inflation_rate_Scenario_2": "___%" ,
  // REQUIREMENTS: MINIMUM VALUE IS -2.00, MAXIMUM VALUE IS 8.00, AND UP TO 2 DECIMAL PLACES ARE ALLOWED
  "Main_considerations_Scenario_2": "..."
  // 2–3 SENTENCES SUMMARIZING THE MAIN CONSIDERATIONS IN THE SCENARIO 2
}
"""

# Survey prompt example (Chinese scenario)
prompt_chinese = """
您正在参与一项关于在线学习平台使用体验的调查研究。
我们希望了解您对在线教育产品的真实看法和使用习惯。

您的个人背景：
您是一位28岁的市场营销专员，本科学历，现居住在上海市浦东新区。
您在一家中等规模的互联网公司工作了5年，月收入约15000元。
您平时工作比较忙碌，但对个人技能提升很重视，经常利用业余时间学习新知识。

您的学习经历：
- 您使用过的在线学习平台：腾讯课堂、网易云课堂、慕课网
- 您主要学习的内容：数字营销、数据分析、职场技能
- 您每周大约花费4-6小时进行在线学习
- 您更喜欢视频+实操的学习方式，不太喜欢纯理论课程

调查问题：
假设现在有一个新的在线学习平台推出，主要特点是：
1. AI个性化学习路径推荐
2. 实时互动答疑功能
3. 学习成果认证和求职匹配
4. 月费制，每月99元

基于您的学习需求和消费能力，请回答以下问题：

请严格按照以下JSON格式输出您的回答：
{
  "platform_interest_level": "非常感兴趣/比较感兴趣/一般/不太感兴趣/完全不感兴趣",
  "most_valuable_features": ["列出您最看重的2-3个功能特点"],
  "pricing_acceptance": "完全可以接受/可以接受/有点贵但可考虑/太贵了/绝对不会付费",
  "usage_frequency_prediction": "每天/每周2-3次/每周1次/每月几次/很少使用",
  "main_concerns": "您对这个新平台最担心的问题是什么",
  "recommendation_likelihood": "一定会推荐/可能会推荐/不确定/不太会推荐/绝对不会推荐"
}
"""

# Build conversation
messages = [
    {"role": "user", "content": prompt_economy}
]

# Generate response
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=2000,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)

# <recall>
# I am a 56-year-old divorced woman living in the Western region of the US. I have a college degree and live in a house with a high market value of $400,000. My political affiliation is Republican, and my annual income is $65,000, which is considered middle-income. I have formed certain beliefs about prices and unemployment based on my experiences and interactions. I tend to believe that prices generally go down and that unemployment will decrease.
# </recall>
# <reasoning>
# Given my personal characteristics and prior expectations, I expect the unemployment rate to decrease and the inflation rate to fall in the coming 12 months. As someone who has experienced economic changes in the past, I believe that when the price of crude oil decreases, it often leads to improved economic conditions, which can result in lower unemployment rates and reduced inflation. Additionally, my political leanings towards the Republican Party, which tends to favor lower taxes and reduced government intervention, might influence my view that economic policies leading to a decrease in oil prices would positively impact the economy.
# </reasoning>
# <final_answer>
# ```json
# {
#   "Unemployment_rate_Scenario_1": "4.00%",
#   "Inflation_rate_Scenario_1": "1.60%",
#   "Unemployment_rate_Scenario_2": "3.00%",
#   "Inflation_rate_Scenario_2": "-1.00%",
#   "Main_considerations_Scenario_2": "I expect the decrease in oil prices to lead to increased economic stability, potentially reducing unemployment and lowering inflation."
# }
# ```  
# </final_answer>
```

### Prompt Construction Guidelines

A complete survey prompt should include:

1. **Role Setting**: Age, gender, occupation, residence, etc.
2. **Economic Status**: Income level, asset status, debt situation, etc.
3. **Educational Background**: Education level, professional background, work experience, etc.
4. **Confidence Level**: Confidence level on relevant topics (extremely weak/weak/moderate/strong/extremely strong)
5. **Life Background**: Family situation, consumption habits, values, etc.
6. **Specific Scenarios**: Detailed hypothetical situations and background information
7. **Question Setup**: Specific questions to be answered
8. **Output Format**: Expected JSON structure and field requirements

### More Examples

#### Chinese Case: Online Learning Platform Usage Experience Survey

```python
# Using prompt_chinese from above example for Chinese survey scenarios
# The model will output responses like:

# <recall>
# 我是一位28岁的市场营销专员，拥有本科学历，现居住在上海。我在这个领域工作了五年，目前月收入约为15000元。我对个人技能的提升非常重视，因此经常利用业余时间学习新知识。根据我的个人背景，我是一个注重实际操作和效果的人，所以我更倾向于视频+实操的学习方式，而不是纯粹的理论课程。这些因素都可能会影响我对在线学习平台的看法和选择。
# </recall>

# <reasoning>
# 对于这个新的在线学习平台，我认为它有几个吸引人的特点：AI个性化学习路径推荐、实时互动答疑功能以及学习成果认证和求职匹配。这些功能对我来说非常重要，因为它们能够帮助我更好地规划学习路径，解决学习过程中遇到的问题，并且证明我的学习成果。此外，学习成果认证和求职匹配也能为我的职业发展提供帮助。考虑到我的消费能力和学习需求，我认为这个月费制的平台是完全可以接受的。
# </reasoning>

# <final_answer>
# ```json
# {
#   "platform_interest_level": "非常感兴趣",
#   "most_valuable_features": ["AI个性化学习路径推荐", "实时互动答疑功能", "学习成果认证和求职匹配"],
#   "pricing_acceptance": "完全可以接受",
#   "usage_frequency_prediction": "每周1次",
#   "main_concerns": "担心平台的安全性和稳定性，以及是否能提供足够的学习资源。",
#   "recommendation_likelihood": "可能会推荐"
# }
# ```  
# </final_answer>
```

## Citation

Research Paper:

```bibtex
@misc{lin2025simulatingmacroeconomicexpectationsusing,
      title={Simulating Macroeconomic Expectations using LLM Agents}, 
      author={Jianhao Lin and Lexuan Sun and Yixin Yan},
      year={2025},
      eprint={2505.17648},
      archivePrefix={arXiv},
      primaryClass={econ.GN},
      url={https://arxiv.org/abs/2505.17648}, 
}
```

If you use the Survey-v1-3B-Beta model in technical implementation, please cite:

```bibtex
@misc{survey-v1-3b-beta,
    title = {Survey-v1-3B-Beta: A Specialized Language Model for Survey Research},
    url = {https://github.com/keta1930/survey-v1-3b-beta},
    author = {keta1930},
    month = {September},
    year = {2025}
}
```

## License

This model follows the Apache 2.0 license.

## Contact Information

For questions or suggestions, please contact through:
- GitHub Issues: https://github.com/keta1930/survey-v1-3b-beta

- Email: keta1930@163.com

---

## Important Disclaimer

**⚠️ Experimental Model Warning**

This model is an experimental research tool. Please note the following important limitations when using:

- **Cannot replace real human respondents**: The survey responses generated by this model are simulation results only and cannot fully represent real population views and behaviors
- **Systematic bias exists**: There may be significant statistical distribution bias between model output and real human survey results
- **Use caution with sensitive topics**: Do not apply this model to topics involving politics, religion, race, gender and other sensitive subjects. The model's generation does not represent real population views
- **Data quality risks**: Research conclusions based on model-generated data may be misleading and should not be directly used for important decisions
- **Ethical compliance responsibility**: Users need to ensure that survey content complies with relevant ethical standards and legal regulations
- **Usage scenario limitations**: This model is suitable for auxiliary survey work and cannot serve as the sole data source for formal research

**Recommended usage scenarios**: Preliminary survey design, questionnaire pretesting, concept validation, and other auxiliary research work. To obtain reliable survey data, traditional human respondent survey methods should still be used.

---