# Survey-v1-3B-Beta

## 简介

Survey-v1-3B-Beta 是一个专门用于问卷调研场景的大语言模型（实验性质，不建议投入真实生产或科研环境），**本模型于《Simulating Macroeconomic Expectations using LLM Agents》([arXiv:2505.17648](https://arxiv.org/abs/2505.17648))研究过程中开发，基于Qwen2.5-3B-Instruct进行训练**，旨在解决传统问卷调研中的效率和成本问题。，专注于探索大语言模型在模拟人类调研行为方面的潜力和局限性。
### 开发背景

传统问卷调研面临诸多挑战：
- **人力成本高昂**：大规模问卷调研需要招募大量受访者，成本巨大
- **样本偏差问题**：难以获得真正代表性的样本群体
- **时间周期长**：从问卷设计到数据收集往往需要数周甚至数月
- **大模型资源消耗**：直接使用deepseek-r1;qwen235b等大规模模型进行问卷调研，API调用成本/部署成本/后训练成本极高

为解决这些问题，我们开发了Survey-v1-3B-Beta，这是一个经过专门优化的3B参数模型，能够以极低的计算成本实现高质量的问卷调研模拟。

### 模型功能

**🎭 多人格模拟**
- 根据年龄、性别、教育程度、收入水平等特征模拟不同受访者
- 支持信心水平调节，影响回答的确定性和倾向性

**🧠 结构化推理**
- 采用三段式回答格式：回忆个人特征 → 推理分析 → 最终答案
- 确保回答的逻辑性和一致性

**📊 标准化输出**
- 严格的JSON格式输出，便于后续数据分析
- 支持数值范围验证和格式约束

**🌍 多语言支持**
- 支持中英文问卷场景
- 适配不同文化背景下的调研需求

### 输出规范

模型采用固定的三段式输出格式：

```xml
<recall>
回忆并陈述与回答相关的个人特征，包括年龄、性别、教育程度、收入水平、居住情况、信心水平等。解释这些个人因素可能如何影响你的观点和回答。
</recall>

<reasoning>
分享你对此场景的自然思考和反应。这可以包括你的直觉感受、个人经历中的相似情况、从他人那里听到的信息，或只是一般印象。无需成为专家或拥有所有事实，只需表达作为普通人对这种情况的看法。
</reasoning>

<final_answer>
json
{
  "字段1": "值1",
  "字段2": "值2",
  ...
}
</final_answer>
```

## 快速开始

### 安装依赖

```bash
pip install transformers torch
```

### 基础使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "/mnt/f/model/survey-r1-3B"  # 替换为实际模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

# 问卷提示词示例（中文场景）
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

# 构建对话
messages = [
    {"role": "user", "content": prompt_chinese}
]

# 生成回答
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

### 提示词构建指南

一个完整的问卷提示词应该包含：

1. **角色设定**：年龄、性别、职业、居住地等
2. **经济状况**：收入水平、资产状况、负债情况等  
3. **教育背景**：学历、专业背景、工作经验等
4. **信心水平**：对相关议题的信心程度（极弱/弱/中等/强/极强）
5. **生活背景**：家庭状况、消费习惯、价值观念等
6. **具体场景**：详细的假设情境和背景信息
7. **问题设置**：具体要回答的问题
8. **输出格式**：期望的JSON结构和字段要求

### 更多示例

#### 英文案例：经济预测调研

```python
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

## 引用

研究论文：

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

如果您在技术实现中使用了Survey-v1-3B-Beta模型，请引用：

```bibtex
@misc{survey-v1-3b-beta,
    title = {Survey-v1-3B-Beta: A Specialized Language Model for Survey Research},
    url = {https://github.com/keta1930/survey-v1-3b-beta},
    author = {keta1930},
    month = {September},
    year = {2025}
}
```

## 许可证

本模型遵循 Apache 2.0 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: https://github.com/keta1930/survey-v1-3b-beta

- Email: keta1930@163.com


---

## 重要声明

**⚠️ 实验性模型警告**

本模型为实验性质的研究工具，使用时请注意以下重要限制：

- **不能替代真实人类受访者**：本模型生成的问卷回答仅为模拟结果，无法完全代表真实人群的观点和行为
- **存在系统性偏差**：模型输出与真实人类调研结果之间可能存在显著的统计分布偏差
- **敏感议题谨慎使用**：请勿在涉及政治、宗教、种族、性别等敏感话题中应用本模型，模型的生成也不代表真实人群的观点
- **数据质量风险**：基于模型生成数据的研究结论可能存在误导性，不应直接用于重要决策
- **伦理合规责任**：使用者需自行确保调研内容符合相关伦理规范和法律法规
- **使用场景限制**：本模型适用于辅助性调研工作，不能作为正式调研的唯一数据来源

**建议使用场景**：初步调研设计、问卷预测试、概念验证等辅助性研究工作。如需获得可靠的调研数据，仍应采用传统的人类受访者调研方法。

---